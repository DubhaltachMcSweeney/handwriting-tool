import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphabet_row_segmentation import (
    _infer_default_punctuation_row,
    default_expected_rows_for_line_count,
    normalize_expected_rows,
    segment_alphabet_rows,
)


def _build_synthetic_row_image(output_path):
    image = Image.new("L", (420, 240), 255)
    draw = ImageDraw.Draw(image)

    layout = [
        ("ABC", 30, 30),
        ("def", 30, 100),
        ("12", 130, 170),
    ]

    width_map = {"A": 20, "B": 22, "C": 18, "d": 18, "e": 16, "f": 14, "1": 12, "2": 18}
    x_gap = 28

    for text, start_x, y in layout:
        x = start_x
        for character in text:
            width = width_map[character]
            draw.rectangle([x, y, x + width, y + 42], fill=0)
            x += width + x_gap

    image.save(output_path)


def _build_synthetic_row_image_with_punctuation(output_path):
    image = Image.new("L", (520, 340), 255)
    draw = ImageDraw.Draw(image)

    layout = [
        ("ABC", 30, 30),
        ("def", 30, 100),
        ("12", 130, 170),
    ]

    width_map = {"A": 20, "B": 22, "C": 18, "d": 18, "e": 16, "f": 14, "1": 12, "2": 18}
    x_gap = 28

    for text, start_x, y in layout:
        x = start_x
        for character in text:
            width = width_map[character]
            draw.rectangle([x, y, x + width, y + 42], fill=0)
            x += width + x_gap

    # punctuation row: period, comma, question mark, exclamation mark
    draw.rectangle([120, 280, 128, 288], fill=0)          # .
    draw.rectangle([170, 282, 178, 290], fill=0)          # , head
    draw.rectangle([176, 290, 182, 300], fill=0)          # , tail
    draw.rectangle([230, 246, 252, 278], fill=0)          # ? body
    draw.rectangle([236, 288, 244, 296], fill=0)          # ? dot
    draw.rectangle([290, 246, 298, 282], fill=0)          # ! body
    draw.rectangle([290, 288, 298, 296], fill=0)          # ! dot

    image.save(output_path)


def test_normalize_expected_rows_collapses_spaces_and_empty_lines():
    rows = normalize_expected_rows("A B C\n\n d e f \n 1 2 ")
    assert rows == ["ABC", "def", "12"]


def test_default_expected_rows_supports_fourth_punctuation_line():
    rows = default_expected_rows_for_line_count(4)
    assert rows == [
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "1234567890",
        ".,?!",
    ]


def test_infer_default_punctuation_row_supports_basic_and_extended_sets():
    assert _infer_default_punctuation_row(4) == ".,?!"
    assert _infer_default_punctuation_row(11) == ".,?!'\":;-()"


def test_segment_alphabet_rows_preserves_row_and_character_order(tmp_path):
    image_path = tmp_path / "alphabet_rows.png"
    _build_synthetic_row_image(image_path)

    rows = segment_alphabet_rows(image_path, expected_text="ABC\ndef\n12")

    assert [len(row) for row in rows] == [3, 3, 2]
    assert "".join(segment.character for segment in rows[0]) == "ABC"
    assert "".join(segment.character for segment in rows[1]) == "def"
    assert "".join(segment.character for segment in rows[2]) == "12"
    assert rows[0][0].line_index == 0
    assert rows[2][1].column_index == 1
    assert isinstance(rows[0][0].raw_image_array, np.ndarray)
    assert rows[0][0].image_array.shape == (28, 28)


def test_segment_alphabet_rows_supports_punctuation_row(tmp_path):
    image_path = tmp_path / "alphabet_rows_punctuation.png"
    _build_synthetic_row_image_with_punctuation(image_path)

    rows = segment_alphabet_rows(image_path, expected_text="ABC\ndef\n12\n.,?!")

    assert [len(row) for row in rows] == [3, 3, 2, 4]
    assert "".join(segment.character for segment in rows[3]) == ".,?!"
