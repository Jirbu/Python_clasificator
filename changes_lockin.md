# Změny vah/prahů pro posílení "lock-in" (identity) v person_manager.py

Tenhle soubor eviduje změny vah a prahů kolem `AppearanceValidator` a
identity-persistence logiky (kdo je "naše" sledovaná osoba), analogicky k
`changes_to_yolo.md`. Cíl: zabránit přeskoku trackeru na jinou osobu v davu
po ztrátě/znovuzískání (LOST → TRACKING).

Formát:
```
### <soubor>:<konstanta>
- Původní hodnota: ...
- Nová hodnota: ...
- Důvod: ...
```

---

## Původní stav (baseline před jakoukoli změnou v tomto souboru)

Zdroj: `person_manager.py`, řádky ~124–132, ověřeno 2026-08-05.

```python
_W_TRACKER             = 0.40   # presence_prob z PersonTracker (temporální stabilita)
_W_KIN                 = 0.15   # kinematic_score = 1 - dist/MAX (blíže = lepší)
_W_MOTION              = 0.20   # (1 - sim_score): vysoká podobnost = statický = penalizace
_W_APPEARANCE          = 0.25   # appearance_score: 1.0 = barva odpovídá historii
_FINAL_THR             = 0.30   # min final_conf pro pipeline SUCCESS

_SCALE_SWITCH_THR      = 0.45   # min scale_err pro detekci přeskoku
_SCALE_SWITCH_APPEAR   = 0.80   # max appearance_score při přeskoku (nízká = jiná osoba)
```

**Zjištěný problém (diagnostikováno na `testovaci_5`/`IMG_7630.MOV`):**
appearance_score je jen měkký příspěvek do váženého průměru (25 % váhy).
Tvrdé zamítnutí kvůli barvě dnes existuje POUZE v `scale_switch` cestě,
která navíc vyžaduje SOUČASNĚ velký skokový rozdíl velikosti postavy
(`scale_err >= 0.45`). Pokud se tracker při LOST re-akvizici v davu chytí
na podobně velkou osobu (žádný scale jump), appearance ho nezastaví vůbec –
matematicky: i appearance_score=0 se remapuje na 0 příspěvku, ale zbylé
složky (tracker+kin+motion, dohromady 75 % váhy) snadno samy o sobě
přehoupnou `_FINAL_THR=0.30`.

Důsledek: 2 ze 4 referenčních skoků na `testovaci_5` nebyly zachyceny,
protože se tracker přepnul na jinou osobu v záběru (uživatel má výrazné
červené tričko, appearance validator by měl umět rozlišit, ale jeho váha
v kombinaci je příliš nízká na to, aby to samo o sobě zabránilo přepnutí).

---

### person_manager.py: `_REACQUIRE_APPEAR_MIN` (nová konstanta)
- Původní hodnota: neexistovalo – appearance_score nikdy netvořil tvrdou brzdu
  mimo `scale_switch` cestu.
- Nová hodnota: `0.45`. Přidán hard-reject při návratu z GHOST/LOST stavu
  (`slot.state in (GHOST, LOST)`), pokud `appearance_validator.has_history`
  a `appearance_score < 0.45` – nezávisle na `scale_err` (na rozdíl od
  `scale_switch`, který vyžaduje SOUČASNĚ velký skokový rozdíl velikosti).
- Důvod: viz "Zjištěný problém" výše. Ověřeno: appearance_validator si drží
  historii přes GHOST i LOST (reset až při LOST→EMPTY timeoutu), takže je
  v okamžiku reakvizice platná.
- **Zjištěno po nasazení:** kryje jen úzkou cestu (formální GHOST/LOST
  přechod). Uživatel nahlásil přeskok mezi osobami (červené triko →
  bílé triko) BEZ zjevného formálního GHOST/LOST přechodu – tedy tenhle
  gate takový případ nekryje. Viz další záznam (dynamický crop margin).

### person_manager.py: `_CROP_MARGIN_MIN`, `_CROP_VEL_SATURATE` (nové konstanty) + `crop_raw_side_ema`
- Původní hodnota: crop margin byl FIXNÍ `_CROP_MARGIN=0.40` vždy, navíc
  `effective_side = max(raw_side, slot.crop_side_ema)` nikdy nezmenšoval crop
  zpátky, jen rostl/stagnoval (EMA se blendovala se `raw_side`, ale samotný
  `_compute_crop` bral vždy jen `max()` z obou).
- Nová hodnota: margin je dynamický podle aktuální rychlosti trackeru
  (`track_info["velocity"]`): `_CROP_MARGIN_MIN=0.10` v klidu, lineárně roste
  k `_CROP_MARGIN=0.40` (max) při rychlosti `>= _CROP_VEL_SATURATE=0.04`
  (norm. jednotky/snímek). Implementováno přes NOVOU odděl. EMA
  `slot.crop_raw_side_ema` (čistě raw bbox, bez marže) – `crop_side_ema`/
  `crop_side_max` zůstávají beze změny (pořád se používají pro ghost-recovery
  crop, který má zůstat generózní).
- Důvod: uživatel při vizuální kontrole `testovaci_5` zjistil, že tracker
  přeskočil na viditelně jinou osobu (odlišná barva trička, jiná vzdálenost/
  velikost) – hypotéza: velký, nikdy se nezmenšující crop v klidové fázi
  snadno pojme i sousední osobu v davu. `_CROP_VEL_SATURATE=0.04` byl
  odhad bez přesné kalibrace na datech.

**VÝSLEDEK: ZAVRŽENO A VRÁCENO ZPĚT (2026-08-05).** Otestováno na všech
4 referenčních videích:

| Video | Efekt |
|---|---|
| IMG_6497 | beze změny (9/9, stejný jako baseline) |
| testovaci_1 | **regrese** – ztratil ref1 (14665-15465), signál úplně zmizel |
| testovaci_4 | podezřelý pokles HIGHLIGHTS clusterů (11→8, bez ground-truth referencí) |
| testovaci_5 | smíšené – získal zpět ref3, ale ztratil dřív spolehlivý ref2 + 3 nové nejasné signály |

Hypotéza proč: crop se zvětšuje na základě rychlosti z PŘEDCHOZÍHO snímku
(jednosnímkové zpoždění) – při náhlém zrychlení (např. rozběh těsně před
saltem) crop dočasně zaostává za skutečnou rychlostí a může postavu lehce
oříznout přesně v kritické fázi. Čistý efekt napříč videi byl negativní,
takže celá změna byla vrácena (`_CROP_MARGIN_MIN`/`_CROP_VEL_SATURATE`/
`crop_raw_side_ema` odstraněny, `effective_side` zpět na
`max(raw_side, slot.crop_side_ema)`). Ponechána jen jako historický
záznam pro případ budoucího opakovaného pokusu s jiným přístupem
(např. look-ahead na akceleraci místo jen aktuální rychlosti, nebo
vyšší minimální margin).

### person_manager.py: `_detect_in_crop`/`_detect_in_crop_hires` – výběr kandidáta v cropu
- Původní chování: `image_detector.detect_all(crop_px)` může vrátit až 2 lidi
  (MediaPipe `num_poses=2`, YOLO `_run_all` vrací všechny detekované) – kód
  ale bral vždy jen `detected[0]`, první v pořadí vráceném modelem, BEZ
  jakékoli pozicové kontroly. Tohle je HLAVNÍ detekční cesta (~94 % snímků).
- Nová hodnota: nová metoda `_pick_crop_candidate()` – při 1 kandidátovi beze
  změny (žádné riziko regrese), při 2+ kandidátech vybere nejbližšího
  `slot.kin_predicted` přes stávající `_nearest_to()` (stejná logika, jaká
  se už používala pro full-frame scan, teď i tady). Všech 5 volání
  `_detect_in_crop(_hires)` teď předává `slot`.
- Důvod: `_nearest_to()` (pozicová/kinematická kontrola, `_W_KIN`+`_W_TRACKER`
  = 55 % váhy finálního skóre) se dosud aplikovala JEN na full-frame scan
  fallback, ne na hlavní crop-based cestu – tam žádná pozicová disambiguace
  neexistovala vůbec. Přímo souvisí s hlášeným přeskokem na `testovaci_5`.
