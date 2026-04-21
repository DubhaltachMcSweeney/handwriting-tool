import csv
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import corrections


def test_save_correction_copies_image_and_writes_csv(tmp_path, monkeypatch):
    corrections_dir = tmp_path / "corrected_samples"
    monkeypatch.setattr(corrections, "CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr(corrections, "IMAGES_DIR", corrections_dir / "images")
    monkeypatch.setattr(corrections, "LABELS_PATH", corrections_dir / "labels.csv")

    image_path = tmp_path / "digit.png"
    Image.new("L", (20, 20), color=255).save(image_path)

    labels_path = corrections.save_correction(
        image_path,
        prediction="3",
        corrected_label="8",
        mode="single",
    )

    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["prediction"] == "3"
    assert rows[0]["corrected_label"] == "8"
    assert Path(rows[0]["saved_image_path"]).exists()
