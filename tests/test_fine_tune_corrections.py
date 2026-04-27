import csv
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fine_tune_corrections import (
    load_labeled_multi_digit_samples,
    load_single_digit_corrections,
)


def test_load_single_digit_corrections_filters_to_valid_single_digit_rows(tmp_path):
    image_path = tmp_path / "digit.png"
    Image.new("L", (20, 20), color=255).save(image_path)

    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "mode",
                "original_path",
                "saved_image_path",
                "prediction",
                "corrected_label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "1",
                "mode": "single",
                "original_path": "a.png",
                "saved_image_path": str(image_path),
                "prediction": "3",
                "corrected_label": "8",
            }
        )
        writer.writerow(
            {
                "timestamp": "2",
                "mode": "multi",
                "original_path": "b.png",
                "saved_image_path": str(image_path),
                "prediction": "31",
                "corrected_label": "31",
            }
        )
        writer.writerow(
            {
                "timestamp": "3",
                "mode": "single",
                "original_path": "c.png",
                "saved_image_path": str(image_path),
                "prediction": "x",
                "corrected_label": "12",
            }
        )

    samples = load_single_digit_corrections(labels_path)

    assert len(samples) == 1
    assert samples[0]["label"] == 8
    assert samples[0]["image_path"] == image_path


def test_load_labeled_multi_digit_samples_uses_filename_labels(tmp_path, monkeypatch):
    image_path = tmp_path / "multi_digit_12.3_001_raw.png"
    Image.new("L", (20, 20), color=255).save(image_path)

    class DummySegment:
        def __init__(self, image_array):
            self.kind = "digit"
            self.image_array = image_array

    dummy_segments = [
        DummySegment([[255]]),
        DummySegment([[255]]),
        DummySegment([[255]]),
    ]

    import fine_tune_corrections

    monkeypatch.setattr(fine_tune_corrections, "segment_symbols", lambda _: dummy_segments)

    samples = load_labeled_multi_digit_samples(tmp_path)

    assert [sample["label"] for sample in samples] == [1, 2, 3]
