"""
video_loader.py
---------------
Modul pro načítání videí a iteraci snímků s frame samplingem.

Zodpovědnost:
  - Otevření video souboru pomocí OpenCV
  - Získání metadat (FPS, počet snímků, rozlišení)
  - Generování snímků s frame skippingem na cílové FPS
  - Výpočet timestamp v milisekundách
"""

import cv2
import os
from pathlib import Path


class VideoLoader:
    """
    Načítá video soubor a poskytuje iterátor snímků s frame samplingem.

    Parametry:
        video_path  -- cesta k video souboru
        target_fps  -- požadovaný počet analyzovaných snímků za sekundu (default 8)
    """

    def __init__(self, video_path: str, target_fps: int = 8):
        self.video_path = video_path
        self.target_fps = target_fps

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Nelze otevřít video: {video_path}")

        # --- metadata videa ---
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Krok pro frame skipping: každý N-tý snímek se analyzuje
        # Např. 30 FPS video, target 8 FPS -> skip každý 3-4. snímek
        # POZNÁMKA: Skipování probíhá zde v loaderu – do person_manageru
        # přichází jen každý frame_step-tý snímek z původního videa.
        self.frame_step = max(1, round(self.video_fps / self.target_fps))

    def get_video_info(self) -> dict:
        """Vrátí základní metadata videa."""
        return {
            "path": self.video_path,
            "fps": self.video_fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "frame_step": self.frame_step,
            "effective_fps": self.video_fps / self.frame_step,
        }

    def calculate_timestamp_ms(self, frame_index: int) -> float:
        """
        Vypočítá timestamp aktuálního snímku v milisekundách od začátku videa.

        frame_index -- absolutní index snímku v původním videu
        """
        return (frame_index / self.video_fps) * 1000.0

    def frame_generator(self):
        """
        Generátor, který yieldi (timestamp_ms, frame, prev_frame) tuplu pro každý
        analyzovaný snímek (tzn. po frame_step krocích).

        prev_frame -- poslední přeskočený snímek těsně PŘED aktuálním zpracovaným
                      snímkem (nebo None pro první snímek). Lze použít jako čistší
                      alternativu k aktuálnímu snímku při hires fallbacku.

        Vypočítává timestamp z absolutního indexu snímku v původním videu,
        aby timestamp přesně odpovídal pozici v čase.
        """
        frame_index = 0
        prev_frame = None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset na začátek

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Zpracuj pouze každý N-tý snímek
            if frame_index % self.frame_step == 0:
                timestamp_ms = self.calculate_timestamp_ms(frame_index)
                yield timestamp_ms, frame, prev_frame
                prev_frame = None  # reset – příští přeskočený snímek přepíše
            else:
                prev_frame = frame  # pamatuj si poslední přeskočený snímek

            frame_index += 1

    def release(self):
        """Uvolní VideoCapture zdroj."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def get_video_files(videos_dir: str) -> list:
    """
    Vrátí seznam všech video souborů v zadané složce.
    Podporované formáty: mp4, avi, mov, mkv.
    """
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos_path = Path(videos_dir)

    if not videos_path.exists():
        raise FileNotFoundError(f"Složka s videi neexistuje: {videos_dir}")

    video_files = [
        str(f)
        for f in sorted(videos_path.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    return video_files
