import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation import segment_digit_arrays, segment_symbols


def test_segment_digit_arrays_splits_left_to_right_digits(tmp_path):
    image_path = tmp_path / "multi_digit.png"
    image = Image.new("L", (220, 100), color=255)
    draw = ImageDraw.Draw(image)

    draw.line([(35, 20), (60, 20), (60, 80)], fill=0, width=8)
    draw.ellipse((130, 20, 190, 80), outline=0, width=8)
    image.save(image_path)

    segments = segment_digit_arrays(image_path)

    assert len(segments) == 2
    assert segments[0].bbox[0] < segments[1].bbox[0]
    assert segments[0].image_array.shape == (28, 28)
    assert segments[1].image_array.shape == (28, 28)


def test_segment_symbols_preserves_one_decimal_dot_between_close_digits(tmp_path):
    image_path = tmp_path / "decimal_digit.png"
    image = Image.new("L", (220, 100), color=255)
    draw = ImageDraw.Draw(image)

    draw.arc((20, 20, 75, 75), start=270, end=90, fill=0, width=8)
    draw.ellipse((96, 70, 106, 80), fill=0)
    draw.line([(130, 20), (130, 80)], fill=0, width=8)
    draw.point([(x, 88) for x in range(0, 220, 10)], fill=0)
    image.save(image_path)

    symbols = segment_symbols(image_path)

    assert [symbol.kind for symbol in symbols] == ["digit", "dot", "digit"]
