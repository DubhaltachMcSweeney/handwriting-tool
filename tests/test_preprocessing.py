import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import preprocess_digit_array, preprocess_digit_image


def _draw_digit_like_image(path, size=(120, 90), background=255, foreground=0):
    image = Image.new("L", size, color=background)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.line(
        [
            (width * 0.25, height * 0.2),
            (width * 0.75, height * 0.2),
            (width * 0.55, height * 0.8),
        ],
        fill=foreground,
        width=max(4, width // 14),
    )
    image.save(path)


def test_preprocess_missing_image_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        preprocess_digit_image(missing_path)


def test_preprocess_invalid_image_raises_value_error(tmp_path):
    bad_image_path = tmp_path / "bad.png"
    bad_image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        preprocess_digit_image(bad_image_path)


@pytest.mark.parametrize("size", [(40, 40), (320, 120), (90, 240)])
def test_preprocess_handles_different_image_sizes(tmp_path, size):
    image_path = tmp_path / "digit.png"
    _draw_digit_like_image(image_path, size=size)

    tensor = preprocess_digit_image(image_path)

    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert tensor.max().item() > 0


def test_preprocess_handles_white_background_black_foreground(tmp_path):
    image_path = tmp_path / "white_bg.png"
    _draw_digit_like_image(image_path, background=255, foreground=0)

    image_array = preprocess_digit_array(image_path)

    assert image_array.shape == (28, 28)
    assert image_array.max() == 255


def test_preprocess_handles_black_background_white_foreground(tmp_path):
    image_path = tmp_path / "black_bg.png"
    _draw_digit_like_image(image_path, background=0, foreground=255)

    image_array = preprocess_digit_array(image_path)

    assert image_array.shape == (28, 28)
    assert image_array.max() == 255


def test_preprocess_existing_single_digit_sample():
    sample_path = PROJECT_ROOT / "samples" / "raw" / "digits" / "digit_3_001_raw.png"

    tensor = preprocess_digit_image(sample_path)

    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert tensor.max().item() > 0
