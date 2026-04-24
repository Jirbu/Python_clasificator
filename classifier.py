"""
classifier.py
-------------
Modul pro klasifikaci akcí z temporálního feature vektoru.

Architektura je záměrně modulární:
  - HeuristicClassifier  -- jednoduché pravidlové prahy, bez trénování (pro testování)
  - RandomForestClassifier -- scikit-learn model (trénující / naučitelný)
  - ActionClassifier      -- fasáda, která lze přepojit na libovolný backend

Pro produkci stačí vyměnit backend za TensorFlow Lite / CoreML wrapper.
"""

import os
import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Třídy akcí ────────────────────────────────────────────────────────────────
ACTION_CLASSES = ["normal", "jump", "acrobatics", "handstand", "spin", "unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTICKÝ KLASIFIKÁTOR (placeholder bez trénování)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicClassifier:
    """
    Jednoduché pravidlové prahy pro initial testing pipeline.
    Nepoužívá strojové učení – rozhoduje na základě klíčových příznaků.

    Struktura feature vektoru (z feature_extractor.py, window_size=6, 53 feat/snímek):
      - Každý snímek v okně má 53 příznaků řazených takto:
          [0:8]   joint angles (8)
          [8:13]  distances    (5)
          [13:17] orientation  (4)
          [17:27] key heights  (10)
          [27:53] motion       (26)

    Heuristiky pracují s průměrem posledního okna.
    """

    def predict(self, temporal_features: np.ndarray) -> str:
        """
        Vrátí label akce pro daný temporal feature vektor.
        """
        # Rozdělíme temporal vektor na snímky (pole window_size × 53)
        n_frames = 6
        feat_per_frame = len(temporal_features) // n_frames
        frames = temporal_features.reshape(n_frames, feat_per_frame)

        # -- Průměrné hodnoty přes okno --
        avg = frames.mean(axis=0)

        # Index příznaků v jednom snímku
        # [0:8] = úhly, [8:13] = vzdálenosti, [13:17] = orientace, [17:27] = výšky
        avg_left_knee_angle  = avg[2]   # levé koleno
        avg_right_knee_angle = avg[3]   # pravé koleno
        torso_tilt_x         = avg[13]  # naklon trupu L-P
        torso_tilt_z         = avg[14]  # naklon trupu F-B
        nose_height          = avg[17]  # výška nosu (neg. = nahoře po normalizaci)
        l_wrist_height       = avg[19]  # výška levého zápěstí
        r_wrist_height       = avg[20]  # výška pravého zápěstí

        # Pohybové příznaky (průměrná velocity klíčových kloubů)
        avg_velocity = frames[:, 27:40].mean()

        # ── Pravidla ──────────────────────────────────────────────────────
        # Stojka: nos je nízko (záporná výška = nad kyčlemi po normalizaci)
        #         a torso je silně nakloněný
        if nose_height < -1.5 and abs(torso_tilt_x) > 30:
            return "handstand"

        # Skok: nos je výrazně výše než normálně a nohy jsou pokrčené
        if nose_height > 0.5 and avg_left_knee_angle < 150 and avg_right_knee_angle < 150:
            return "jump"

        # Akrobacie: velký naklon trupu + vysoká velocity
        if abs(torso_tilt_z) > 45 and avg_velocity > 0.3:
            return "acrobatics"

        # Spin: zápěstí jsou vysoko a pohyb je rychlý
        if l_wrist_height < -0.5 and r_wrist_height < -0.5 and avg_velocity > 0.4:
            return "spin"

        return "normal"

    def predict_proba(self, temporal_features: np.ndarray) -> dict:
        """Vrátí přibližné pravděpodobnosti (binary 0/1 pro heuristiku)."""
        label = self.predict(temporal_features)
        return {cls: (1.0 if cls == label else 0.0) for cls in ACTION_CLASSES}


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM FOREST KLASIFIKÁTOR
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel:
    """
    scikit-learn RandomForestClassifier obalený do standardního rozhraní.
    Lze natrénovat na labeled datech a pak persistovat na disk.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = _RFC(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,          # využije všechna CPU jádra
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ACTION_CLASSES)
        self._trained = False

    def train(self, X_train: np.ndarray, y_train: list) -> None:
        """
        Natrénuje model.

        X_train -- matice (N, temporal_features)
        y_train -- list stringových labelů
        """
        y_encoded = self.label_encoder.transform(y_train)
        self.model.fit(X_train, y_encoded)
        self._trained = True
        logger.info("RandomForest natrénován na %d vzorcích.", len(y_train))

    def predict(self, temporal_features: np.ndarray) -> str:
        if not self._trained:
            logger.warning("Model není natrénován – vracím 'unknown'.")
            return "unknown"
        pred_encoded = self.model.predict(temporal_features.reshape(1, -1))
        return self.label_encoder.inverse_transform(pred_encoded)[0]

    def predict_proba(self, temporal_features: np.ndarray) -> dict:
        if not self._trained:
            return {cls: (1.0 if cls == "unknown" else 0.0) for cls in ACTION_CLASSES}
        proba = self.model.predict_proba(temporal_features.reshape(1, -1))[0]
        classes = self.label_encoder.inverse_transform(self.model.classes_)
        return {cls: float(p) for cls, p in zip(classes, proba)}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "encoder": self.label_encoder}, f)
        logger.info("Model uložen: %s", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoder = data["encoder"]
        self._trained = True
        logger.info("Model načten: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# FASÁDA – ActionClassifier
# ─────────────────────────────────────────────────────────────────────────────

class ActionClassifier:
    """
    Jednotné rozhraní na klasifikátor.

    Chování:
      - Pokud model_path=None nebo soubor neexistuje, použije HeuristicClassifier.
      - Pokud model_path existuje, načte RandomForestModel ze souboru.
      - Lze kdykoli zavolat `load_model()` nebo `set_backend()` a vyměnit backend.

    Toto rozhraní je připraveno pro budoucí výměnu za CoreML/TFLite wrapper.
    """

    def __init__(self, model_path: str | None = None):
        if model_path and os.path.exists(model_path):
            rf = RandomForestModel()
            rf.load(model_path)
            self._backend = rf
            logger.info("Klasifikátor načten z: %s", model_path)
        else:
            self._backend = HeuristicClassifier()
            logger.info("Klasifikátor: HeuristicClassifier (placeholder).")

    def predict(self, temporal_features: np.ndarray) -> str:
        """Vrátí string label akce."""
        return self._backend.predict(temporal_features)

    def predict_proba(self, temporal_features: np.ndarray) -> dict:
        """Vrátí dict {label: probability}."""
        return self._backend.predict_proba(temporal_features)

    def set_backend(self, backend) -> None:
        """Výměna backendu za libovolný objekt s metodami predict() a predict_proba()."""
        self._backend = backend

    def load_model(self, model_path: str) -> None:
        """Načte RandomForest model ze souboru a nastaví ho jako backend."""
        rf = RandomForestModel()
        rf.load(model_path)
        self._backend = rf
