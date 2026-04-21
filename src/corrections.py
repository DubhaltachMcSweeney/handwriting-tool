import csv
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = PROJECT_ROOT / "corrected_samples"
IMAGES_DIR = CORRECTIONS_DIR / "images"
LABELS_PATH = CORRECTIONS_DIR / "labels.csv"

FIELDNAMES = [
    "timestamp",
    "mode",
    "original_path",
    "saved_image_path",
    "prediction",
    "corrected_label",
]


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_image(image_path, timestamp):
    image_path = Path(image_path)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved_image_path = IMAGES_DIR / f"{image_path.stem}_{timestamp}{image_path.suffix}"
    shutil.copy2(image_path, saved_image_path)
    return saved_image_path


def save_correction(image_path, prediction, corrected_label, mode="single"):
    timestamp = _timestamp()
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    saved_image_path = _copy_image(image_path, timestamp)

    row = {
        "timestamp": timestamp,
        "mode": mode,
        "original_path": str(Path(image_path)),
        "saved_image_path": str(saved_image_path),
        "prediction": str(prediction),
        "corrected_label": str(corrected_label),
    }

    file_exists = LABELS_PATH.exists()
    with LABELS_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return LABELS_PATH
