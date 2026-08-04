# Změny prahů při přechodu z MediaPipe na YOLOv8-pose

Tenhle soubor eviduje KAŽDOU hodnotu/konstantu v `person_manager.py` (a
souvisejících souborech), která byla původně vyladěná na MediaPipe škálu
jistoty a teď se mění kvůli YOLOv8-pose. U každé položky je uvedena původní
(MediaPipe) hodnota, ať se dá v případě potřeby vrátit zpět.

Formát:
```
### <soubor>:<konstanta/proměnná>
- Původní (MediaPipe) hodnota: ...
- Nová hodnota: ...
- Důvod: ...
```

---

### person_manager.py: `PoseConsistencyValidator.suspicious_thr` (přes `_SUSPICIOUS_THR_BY_MODEL`)
- Původní (MediaPipe) hodnota: `3.0` (natvrdo, konstruktor bez argumentu)
- Nová hodnota: `6.0` pro `POSE_MODEL == "yolov8"` (mediapipe zůstává 3.0)
- Důvod: v testu na `IMG_6497.MOV` byl `pose_suspicious` naprosto dominantní příčina
  backup fallbacku (267 z ~270 spuštění). Validátor srovnává aktuální frame-to-frame
  změnu délek segmentů/úhlů s klouzavým průměrem posledních snímků – to je z principu
  citlivé na JAKÝKOLIV náhlý rychlý pohyb (ne jen na chybu detekce), takže začátek
  salta samo o sobě snadno vypadá "podezřele". Empiricky odzkoušeno (viz konverzace).

### person_manager.py: `PoseValidator.min_torso_height` (přes `_POSE_GEO_MIN_TORSO_BY_MODEL`)
- Původní (MediaPipe) hodnota: `_POSE_GEO_MIN_TORSO = 0.03`
- Nová hodnota: `0.015` pro `POSE_MODEL == "yolov8"` (stejná jako stávající "relaxed" režim
  validátoru, mediapipe zůstává 0.03)
- Důvod: diagnostika na 9 referenčních oknech `IMG_6497.MOV` ukázala, že "pose_geo" (L2
  geometrická vrstva) je dominantní příčinou odmítnutí (23 z crop-stage + 17 z fullframe
  pokusů, z ~153 vzorků). Torso_height (2D projekce trupu) se při rotaci/salte přirozeně
  zkracuje kvůli zkrácení perspektivou – uvolnění prahu na hodnotu, kterou kód už jinak
  používá pro "relaxed" stavy (GHOST/LOST), by mělo tyhle případy propustit.
  (Pozn.: empiricky se ukázalo, že tohle NENÍ skutečná dominantní příčina –
  viz další záznam. Ponecháno beze změny efektu, ale hodnota zůstává uvolněná.)

### person_manager.py: `PoseValidator.min_shoulder_width` a `min_hip_width` (přes `_POSE_GEO_MIN_WIDTH_BY_MODEL`)
- Původní (MediaPipe) hodnota: `0.01` pro oba (natvrdo v `PoseValidator.__init__`, nikdy
  nebyly explicitně předávány z `person_manager.py`)
- Nová hodnota: `0.0005` pro `POSE_MODEL == "yolov8"` (mediapipe zůstává 0.01)
- Důvod: **tohle byla skutečná dominantní příčina** L2 geometrických odmítnutí (746 z 754
  zachycených selhání = 99 %, ověřeno dočasnou instrumentací přímo v `pose_validator.py`).
  Šíře ramen (`shoulder_width`) i boků (`hip_width`) klesají téměř na nulu (naměřeno
  0.0008–0.0084), když se tělo během rotace/salta promítá skoro z boku – zatímco
  `torso_height` byla ve VŠECH těchto případech v pořádku (0.09–0.11). Kód už měl OR
  logiku pro přesně tenhle scénář ("propustí boční záběry kde jedno z nich degeneruje"),
  ale práh 0.01 byl příliš přísný na to, aby OBĚ hodnoty prošly současně u tohohle
  konkrétního modelu/pohybu.

### person_manager.py: `_MOTION_HARD_THR_CROP` a `_MOTION_HARD_THR_FULL` (přes `*_BY_MODEL`)
- Původní (MediaPipe) hodnota: `_MOTION_HARD_THR_CROP = 0.29`, `_MOTION_HARD_THR_FULL = 0.40`
- Nová hodnota: `0.10` / `0.15` pro `POSE_MODEL == "yolov8"` (mediapipe zůstává 0.29/0.40)
- Důvod: po opravě geometrických prahů se ukázalo, že motion hard-threshold je NOVÝ
  dominantní blokující bod – na 9 krátkých referenčních klipech (extrahovaných kolem
  každého referenčního času) odmítal 145/297 snímků (49 %). Naměřené `1-sim` hodnoty byly
  těsně pod 0.29 (0.18–0.29), ne blízko nuly (což by značilo opravdu statický snímek).
  `sim_score` se počítá z obrazových pixelů (fázová korelace + NCC na ROI), ne z pozic
  landmarků – takže nejde o čistě "YOLOv8 vs MediaPipe" jev, spíš tenhle práh byl asi
  vždy moc přísný na rychlou akrobacii při 8fps vzorkování a jen se to dřív neprojevilo
  (tyhle snímky dřív padaly už na dřívějších (geometrických) vrstvách).
