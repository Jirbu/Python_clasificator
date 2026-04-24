# Implementační plán - Pipeline pro analýzu lidského pohybu

## 🎯 Celkový přehled

Vytvoření Python pipeline pro analýzu akcí z videí s využitiem pose detection, temporální analýzy a machine learning klasifikace.

---

## 📋 Implementační kroky

### KROK 1: Příprava prostředí a závislostí
**⏱️ Odhadovaná doba: 30 minut**

#### Úkoly:
- [ ] Vytvoření virtual environment
- [ ] Instalace závislostí (opencv-python, mediapipe, numpy, pandas, scikit-learn)
- [ ] Vytvoření základní struktury projektu
- [ ] Příprava testovacích složek `/videos/` a `/output/`

#### Deliverable:
- Funkční vývojové prostředí
- requirements.txt soubor
- Základní struktura projektu

---

### KROK 2: Video Loader modul
**⏱️ Odhadovaná doba: 1 hodina**

#### Funkcionality:
- [ ] Načítání videí z `/videos/` složky pomocí OpenCV  
- [ ] Získání video metadat (FPS, frame count, rozlišení)
- [ ] Frame iterátor s podporou frame sampling
- [ ] Implementace efektivního frame skippingu

#### Klíčové komponenty:
```python
class VideoLoader:
    def __init__(self, video_path, target_fps=8)
    def get_video_info(self)
    def frame_generator(self)
    def calculate_timestamp_ms(self, frame_index)
```

#### Deliverable:
- `video_loader.py` s kompletní funkcionalitou
- Unit testy pro validation

---

### KROK 3: Pose Detector modul  
**⏱️ Odhadovaná doba: 1.5 hodiny**

#### Funkcionality:
- [ ] Inicializace MediaPipe Pose
- [ ] Frame preprocessing (resize na 256x144)
- [ ] Detekce 33 landmarks na každém snímku
- [ ] Extrakce x, y, z, visibility hodnot
- [ ] Error handling pro chybějící pose detekci

#### Klíčové komponenty:
```python
class PoseDetector:
    def __init__(self, confidence_threshold=0.5)
    def preprocess_frame(self, frame)
    def detect_pose(self, frame)
    def extract_landmarks(self, pose_results)
```

#### Deliverable:
- `pose_detector.py` s robustní pose detection
- Validace na testovacích snímcích

---

### KROK 4: Feature Extractor modul
**⏱️ Odhadovaná doba: 2-3 hodiny**  

#### Funkcionality:
- [ ] **Landmark normalizace:**
  - Výpočet hip_center
  - Translace všech bodů k hip_center  
  - Scale normalizace pomocí torso délky
  - Volitelná rotační normalizace

- [ ] **Joint angles extraction:**
  - Úhly loktů, kolen, kyčlí, ramen
  - Implementace angle calculation mezi vektory

- [ ] **Distance features:**  
  - Vzdálenost rukou, nohou
  - Šířka ramen
  - Výška postavy

- [ ] **Motion features:**
  - Rychlost kloubů (velocity)
  - Zrychlení kloubů (acceleration)  
  - Body orientation (torso tilt)

#### Klíčové komponenty:
```python
class FeatureExtractor:
    def normalize_landmarks(self, landmarks)
    def calculate_joint_angles(self, landmarks)
    def calculate_distances(self, landmarks)  
    def calculate_motion_features(self, current_landmarks, previous_landmarks)
    def extract_features(self, landmarks, previous_landmarks=None)
```

#### Deliverable:
- `feature_extractor.py` s ~40-80 features per frame
- Feature validation a visualizace

---

### KROK 5: Temporal Model modul
**⏱️ Odhadovaná doba: 1 hodina**

#### Funkcionality:
- [ ] Sliding window buffer pro posledních 6 snímků
- [ ] Concatenation feature vektorů (6 × features_per_frame)
- [ ] Buffer management (add, get, is_ready)
- [ ] Handling počátečních snímků (< 6 frames)

#### Klíčové komponenty:
```python
class TemporalWindow:
    def __init__(self, window_size=6)
    def add_frame_features(self, features)
    def get_temporal_features(self)
    def is_ready(self)
    def reset(self)
```

#### Deliverable:
- `temporal_model.py` s buffer management
- Unit testy pro temporal logic

---

### KROK 6: Classifier modul
**⏱️ Odhadovaná doba: 2 hodiny**

#### Funkcionality:
- [ ] **Baseline klasifikátor:**
  - RandomForestClassifier (scikit-learn)
  - Jednoduchá heuristická pravidla pro testing

- [ ] **Prediction interface:**
  - Classes: normal, jump, acrobatics, handstand, spin, unknown
  - Confidence scores
  - Fallback logika

- [ ] **Model management:**
  - Load/save trained models  
  - Készní pro výměnu za neural network

#### Klíčové komponenty:
```python
class ActionClassifier:
    def __init__(self, model_path=None)
    def train(self, X_train, y_train)
    def predict(self, features)
    def predict_proba(self, features)  
    def save_model(self, path)
    def load_model(self, path)
```

#### Deliverable:
- `classifier.py` s modulární structured
- Mock model pro testing

---

### KROK 7: Main Pipeline Integration
**⏱️ Odhadovaná doba: 2 hodiny**

#### Funkcionality:
- [ ] **Pipeline orchestration:**
  - Integrace všech modulů
  - Video processing loop
  - Error handling a logging

- [ ] **Batch processing:**
  - Processing všech videí v `/videos/`
  - Parallel processing možnosti
  - Progress tracking

- [ ] **Output generation:**
  - Continuous CSV writing
  - Format: timestamp_ms, action
  - Output file management

#### Klíčové komponenty:
```python
class MainPipeline:
    def __init__(self, videos_dir, output_dir)
    def process_video(self, video_path)
    def process_all_videos(self)
    def setup_logging(self)
```

#### Deliverable:
- `main.py` s kompletní pipeline
- CSV output validation

---

### KROK 8: Testing & Optimization
**⏱️ Odhadovaná doba: 1-2 hodinas**

#### Úkoly:
- [ ] **Performance testing:**
  - Memory usage profiling
  - Processing speed benchmarks
  - Bottleneck identification

- [ ] **Integration testing:**
  - End-to-end pipeline testing
  - Various video formats validation
  - Error case handling

- [ ] **Code quality:**
  - Code review a refactoring
  - Documentation dokončení
  - Configuration externalizace

#### Deliverable:
- Performance report
- Test coverage
- Optimalizované výkon

---

## 🎯 Milníky a kritéria úspěchu

### Milestone 1: Core Pipeline (Kroky 1-4)
**Kritéria úspěchu:**
- ✅ Video se úspěšně načítá a zpracovává
- ✅ Pose detection funguje na testovacích videích  
- ✅ Features se extrahuji v správném formátu
- ✅ Normalizace produkuje konzistentní výsledky

### Milestone 2: Temporal & Classification (Kroky 5-6)  
**Kritéria úspěchu:**
- ✅ Temporal window správně aggreguje features
- ✅ Klasifikátor vrací validente predictions
- ✅ Modular design připraven pro future improvements

### Milestone 3: Production Ready (Kroky 7-8)
**Kritéria úspěchu:**
- ✅ Pipeline zpracuje všechna videa v batch módu
- ✅ CSV output je validní a čitelný
- ✅ Performance je akceptovatelné (< 2x real-time)
- ✅ Kód je well-documented a maintainable

---

## 🔧 Teknické poznámky

### Performance considerations:
- Frame sampling critical pro performance (30→8 FPS)
- Memory management pro feature buffers
- Možnost paralelizace pro batch processing

### Future-proofing:  
- Abstracted interfaces pro snadnou migraci na C++/Swift
- Configurable parameters (FPS, features, window size)  
- Model replacement interface

### Dependencies:
```
opencv-python>=4.5.0
mediapipe>=0.8.0  
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

Tento plán vám poskytuje strukturovaný přístup k implementaci s jasnými milníky a deliverables. Každý krok je navržen tak, aby byl testovatelný a nezávislý na ostatních komponentech.