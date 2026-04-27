import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from letter_corrections import align_characters, normalize_text_characters, save_letter_text_corrections
from text_segmentation import TextSegment


def test_normalize_text_characters_keeps_letters_and_period():
    assert normalize_text_characters("Alice was.\n") == list("ALICEWAS.")


def test_align_characters_handles_insertions_and_deletions():
    alignment = align_characters(list("ALOB"), list("ALICE."))
    assert alignment[0] == (0, 0)
    assert alignment[1] == (1, 1)
    assert any(item[0] is None or item[1] is None for item in alignment)


def test_save_letter_text_corrections_saves_aligned_character_samples(tmp_path, monkeypatch):
    image_path = tmp_path / "sentence.png"
    image_path.write_bytes(b"fake")

    dummy_segments = [
        [
            TextSegment(index=0, bbox=(0, 0, 10, 10), image_array=np.full((28, 28), 255, dtype=np.uint8)),
            TextSegment(index=1, bbox=(12, 0, 10, 10), image_array=np.full((28, 28), 255, dtype=np.uint8)),
            TextSegment(index=2, bbox=(24, 0, 10, 10), image_array=np.full((28, 28), 255, dtype=np.uint8)),
        ]
    ]

    import letter_corrections

    monkeypatch.setattr(letter_corrections, "CORRECTIONS_DIR", tmp_path / "letter_segments")
    monkeypatch.setattr(letter_corrections, "IMAGES_DIR", tmp_path / "letter_segments" / "images")
    monkeypatch.setattr(letter_corrections, "LABELS_PATH", tmp_path / "letter_segments" / "labels.csv")
    monkeypatch.setattr(letter_corrections, "segment_text_lines", lambda _: dummy_segments)

    labels_path, saved_count, segment_count, corrected_count = save_letter_text_corrections(
        image_path,
        predicted_text="ABC",
        corrected_text="ADC",
    )

    assert labels_path.exists()
    assert saved_count == 3
    assert segment_count == 3
    assert corrected_count == 3

    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert [row["corrected_char"] for row in rows] == ["A", "D", "C"]
