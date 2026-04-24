# Human Action Recognition Pipeline

Detekce a klasifikace pohybů z videí pomocí pose detection (MediaPipe) a temporální analýzy.

## Spuštění

```bash
pip install -r requirements.txt
python main.py
```

## Volitelné parametry

```bash
python main.py --videos ./videos --output ./output --fps 8 --model ./models/rf_model.pkl
```

| Parametr  | Popis                                 | Default      |
|-----------|---------------------------------------|--------------|
| --videos  | Složka se vstupními videi             | ./videos     |
| --output  | Složka pro výstupní CSV               | ./output     |
| --fps     | Cílové FPS pro analýzu                | 8            |
| --model   | Cesta k .pkl natrénovaného modelu     | None         |

## Výstupní CSV

```
timestamp_ms,action
0,normal
125,normal
250,jump
375,acrobatics
```

## Pipeline

```
VIDEO → FRAME SAMPLING → RESIZE → POSE DETECTION → NORMALIZACE
→ FEATURE EXTRACTION → TEMPORAL WINDOW (6 snímků) → KLASIFIKACE → CSV
```

## Klasifikované akce

- `normal` – normální chůze/stání
- `jump` – skok
- `acrobatics` – akrobacie
- `handstand` – stojka
- `spin` – rotace/piroueta
- `unknown` – neznámá akce

## Budoucí vývoj

Projekt je připraven pro migraci na mobilní platformy:
- **iOS**: Swift + CoreML
- **Android**: C++ + TensorFlow Lite
