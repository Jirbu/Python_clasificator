# Projekt: Mobilní aplikace pro analýzu lidského pohybu

## Přehled projektu

**Hlavní cíl:** Vytvořit mobilní aplikaci pro detekci a klasifikaci speciálních pohybů (akrobacie, skoky, neobvyklé pozice) z videí pomocí analýzy kostry těla.

## Architektura systému

### Současná fáze (Python prototyp)
- **Účel:** Test algoritmu a optimalizace
- **Technologie:** Python + MediaPipe + OpenCV + scikit-learn
- **Prostředí:** Desktop/server pro vývoj a testování

### Cílová fáze (Mobilní aplikace)
- **iOS verze:** Swift + Metal/CoreML pro výpočetně náročné operace
- **Android verze:** C++/Kotlin + OpenGL/TensorFlow Lite
- **React wrapper:** Existující React aplikace jako UI wrapper

## Technické specifikace

### Pipeline zpracování
1. **Video loading** - načtení videa
2. **Frame sampling** - vzorkování z 30 FPS na 8 FPS  
3. **Resize** - změna rozlišení na 256x144
4. **Pose detection** - MediaPipe Pose (33 landmarks)
5. **Landmark normalization** - normalizace kostry
6. **Feature extraction** - extrakce příznaků (40-80 features)
7. **Temporal window** - sliding window posledních 6 snímků
8. **Action classification** - klasifikace akcí
9. **CSV output** - výstup s timestamp + akce

### Klasifikace akcí
- normal
- jump  
- acrobatics
- handstand
- spin
- unknown

### Výpočetní požadavky
- **Frame rate:** 8 FPS (efektivní)
- **Rozlišení:** 256x144 px
- **Temporální okno:** 6 snímků
- **Feature vektor:** ~6 × (40-80) = 240-480 dimenzí

## Struktura souborů

### Python prototyp
```
/videos/           # vstupní videa
/output/           # výstupní CSV soubory
video_loader.py    # načítání videí
pose_detector.py   # MediaPipe Pose detection
feature_extractor.py # extrakce příznaků
temporal_model.py  # temporální model
classifier.py      # klasifikátor akcí
main.py           # hlavní script
```

### Budoucí mobilní struktura
- **Core algoritmus:** C++/Swift moduly
- **UI:** React Native wrapper (existující)
- **Modely ML:** TensorFlow Lite / CoreML

## Klíčové pozorování

### Optimalizace pro mobilní prostředí
1. **Paměťové nároky** - minimalizovat kopírování dat
2. **Výpočetní složitost** - optimalizovat algoritmy pro ARM procesory
3. **Energetická efektivita** - využít hardware akceleraci (GPU/NPU)
4. **Real-time processing** - minimální latence

### Migrace strategie
1. **Fáze 1:** Python prototyp (současnost)
2. **Fáze 2:** Optimalizace algoritmů
3. **Fáze 3:** Převod do C++/Swift
4. **Fáze 4:** Integrace s React Native

## Potenciální výzvy

### Python → Mobilní migrace
- Závislost na MediaPipe (potřeba alternatív pro mobilní)
- scikit-learn modely → TensorFlow Lite konverze
- OpenCV optimalizace pro ARM

### Výkonové úvahy
- Real-time zpracování vs. batch processing
- On-device vs. cloud processing
- Model velikost vs. přesnost

## TODO ve vývoji
- [ ] Implementovat Python pipeline
- [ ] Testovat na různých videích
- [ ] Optimalizovat feature extraction
- [ ] Vytvořit tréninková data
- [ ] Natrénovat lepší klasifikátor
- [ ] Připravit migrační strategii

## Poznámky k implementaci
- Kód musí být modulární pro snadnou migraci
- Všechny konstanty vynést do konfiguračního souboru
- Připravit benchmark testy pro pozdější optimalizaci
- Dokumentovat všechny preprocessing kroky pro replikaci na mobilních platformách