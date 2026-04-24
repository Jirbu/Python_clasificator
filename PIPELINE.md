# Průběh zpracování jednoho snímku (frame 50)

Předpoklad: vše inicializováno, předchozí snímky zpracovány, buffery plné.

---

## 1. Načtení snímku

`VideoLoader` vydá frame a `timestamp_ms`. Přeskakuje snímky tak, aby výsledek byl ~8 FPS.

---

## 2. MultiPersonManager.update()

Hlavní koordinátor. Zpracuje Slot 0 (Person 1) a Slot 1 (Person 2) nezávisle.

### 2a. Slot 0 – Person 1

**Podle aktuálního stavu slotu:**

- **TRACKING** → `PoseDetector` (VIDEO mode) zpracuje celý frame. MediaPipe vrátí (33,4) landmarks nebo None.
- **LOST** → `PoseDetectorImage` (IMAGE mode) zpracuje pouze `frozen_crop` oblast. Pokud najde pózu, přepočítá souřadnice na full-frame.

#### _run_pipeline(slot0, frame, raw_lm)

1. **PoseValidator** zkontroluje landmarks:
   - L1: dostatečná viditelnost klíčových kloubů (ramena, kyčle)
   - L2: geometrie (poměr šířky ramen, pozice kloubů)
   - Výstup: `valid_pose` (bool) + případně `invalid_reason`

2. **MotionValidator** dostane celý frame a landmarks (pokud valid):
   - Přidá frame do kruhového bufferu (256×144 grayscale)
   - `phaseCorrelate` na celém 256×144 snímku vůči historickým snímkům → globální posun kamery (dx, dy)
   - Pokud posun > 2 px → `warpAffine` historický frame (kompenzace otřesu kamery)
   - Pro každou část těla (torso, levá/pravá paže, levá/pravá noha): extrahuje ROI ze `zarovnaného` historického framu, spočítá NCC podobnost
   - Vážená agregace: `sim_score`
   - Pokud `sim_score >= 0.71` → `region_dynamic = False` (osoba nestojí, ale nestýká se)
   - Výstup: `region_dynamic` (bool), `sim_score`, `motion_info`

3. **PersonTracker** dostane `valid_pose=motion_valid` (AND podmínka):
   - EMA vyhlazení presence_prob
   - State machine: NO_PERSON / PERSON_UNCERTAIN / PERSON_PRESENT
   - Grace period (3 snímky bez detekce → stále PRESENT)
   - Ghost tracking (5 snímků predikce pozice kinematikou)
   - Reacquire mode po vypršení ghost: příští detekce musí prokázat pohyb
   - Výstup: `tracker_present` (bool), `track_info` (pozice, rychlost, ghost, stav)

4. `person_present = tracker_present AND region_dynamic`

#### _update_state(slot0, result)

- Rozhoduje o stavu slotu (TRACKING / LOST / EMPTY) **podle tracker stavu**, ne podle `person_present`
- TRACKING→LOST pouze když tracker=NO_PERSON a ghost=False (nepřehodí kvůli pohybu)
- Při přechodu TRACKING→LOST: zapamatuje `frozen_crop`, resetuje tracker + PoseValidator
- Ve stavu LOST: inkrementuje `lost_frames`; po 80 snímcích (10 s) → EMPTY

---

### 2b. Slot 1 – Person 2

**Pokud EMPTY:** vrátí prázdný výsledek, přeskočí.

**Pokud TRACKING nebo LOST:** `PoseDetectorImage` zpracuje `crop` nebo `frozen_crop` oblast (IMAGE mode). Souřadnice přepočítá na full-frame.

`_run_pipeline(slot1, frame, raw_lm, bypass_motion=True)` — stejný průběh jako Slot 0, ale motion výsledek se ignoruje (`region_dynamic` vždy True).

---

### 2c. Full-frame scan (hledání Person 2)

`PoseDetectorImage` zpracuje celý frame (IMAGE mode, max 2 pózy).

`_handle_scan()` pro každou detekci:
- Ignoruje detekce blíže než 0.30 normalize dist od Person 1
- Zkontroluje viditelnost klíčových kloubů (ramena, kyčle) > 0.65
- Sleduje kandidáta přes čas: 3 po sobě jdoucí snímky na stejném místě → Person 2 potvrzena → Slot 1 přejde do TRACKING

---

## 3. main.py – rozhodovací logika

```
results, slot0_lost = multi_manager.update(...)
r0 = results[0]   # Person 1
r1 = results[1]   # Person 2
```

**Pokud `slot0_lost=True`:** reset `TemporalWindow` a `JumpDetector` (nezačínáme klasifikovat se starými daty).

**Pokud `person_present=False`:** zapiš debug frame, přeskoč klasifikaci (`continue`).

**Pokud `person_present=True` ale `valid_pose=False`:** ghost frame – tracker říká "osoba tu je", ale MediaPipe ji nevidí. Zapiš debug frame, přeskoč (`continue`).

---

## 4. Extrakce příznaků (Person 1)

`FeatureExtractor.extract_features(landmarks)` → vektor příznaků:
- Úhly kloubů (lokty, kolena, kyčle, ramena)
- Normalizované vzdálenosti (např. šířka ramen jako měřítko)
- Výška těžiště

---

## 5. Temporální okno

`TemporalWindow.add_frame_features(features)` – uloží příznaky do klouzavého okna (6 snímků).

Pokud okno ještě není plné (warm-up) → přeskoč klasifikaci.

---

## 6. JumpDetector

`JumpDetector.update(frame, landmarks, timestamp_ms)` – fyzikální validace skoku:
- Sleduje výšku ramen v čase
- Detekuje charakteristický pohyb nahoru-dolů

---

## 7. Klasifikace

`ActionClassifier.predict(temporal_features)` → akce (string, např. "jump", "walk", "stand")

`JumpDetector.combine_with_classifier(action, physics_is_jump)` → může přepsat akci pokud fyzika detekovala skok a klasifikátor ne.

---

## 8. Výstup

- **CSV:** zapíše `timestamp_ms, akce`
- **Debug video:** `DebugVisualizer.write_frame()`:
  - Person 1 skeleton + label "P1" na střed torsa (zelená)
  - Person 2 skeleton + label "P2" na střed torsa (žlutá) pokud aktivní
  - Crop box Person 2 (čárkovaný, cyan=TRACKING, oranžová=LOST)
  - Debug panel: presence_prob, sim_score, tracker stav, akce
  - Motion ROI overlay (torso bounding box)
