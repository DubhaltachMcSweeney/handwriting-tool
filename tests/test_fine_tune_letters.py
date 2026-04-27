import csv
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fine_tune_letters import load_raw_letter_samples, load_segment_letter_corrections


def test_load_segment_letter_corrections_reads_aligned_samples(tmp_path):
    image_path = tmp_path / "segment.png"
    Image.new("L", (20, 20), color=255).save(image_path)

    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "original_path",
                "segment_index",
                "saved_image_path",
                "predicted_char",
                "corrected_char",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "1",
                "original_path": "sentence.png",
                "segment_index": "1",
                "saved_image_path": str(image_path),
                "predicted_char": "A",
                "corrected_char": "B",
            }
        )

    samples = load_segment_letter_corrections(labels_path)

    assert len(samples) == 1
    assert samples[0]["label"] == 1


def test_load_raw_letter_samples_uses_filename_labels(tmp_path):
    image_path = tmp_path / "letter_q_001_raw.png"
    Image.new("L", (20, 20), color=255).save(image_path)

    samples = load_raw_letter_samples(tmp_path)

    assert len(samples) == 1
    assert samples[0]["label"] == ord("Q") - ord("A")
