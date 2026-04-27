import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from render_handwriting import render_handwriting


def _create_character_sample(path, character_text):
    image = Image.new("RGBA", (80, 120), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((20, 30), character_text, fill=(0, 0, 0, 255))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path)
    else:
        image.save(path)


def test_render_handwriting_creates_output_image_with_digits_and_letters(tmp_path):
    digits_root = tmp_path / "font_digits"
    for digit in ["1", "2", "3"]:
        digit_dir = digits_root / digit
        digit_dir.mkdir(parents=True)
        _create_character_sample(digit_dir / f"digit_{digit}_001.png", digit)

    letters_root = tmp_path / "font_letters"
    for case_folder, letter in [("uppercase", "A"), ("lowercase", "b")]:
        letter_dir = letters_root / case_folder / letter
        letter_dir.mkdir(parents=True)
        _create_character_sample(letter_dir / f"letter_{letter}_001.png", letter)

    output_path = tmp_path / "rendered.png"
    rendered_path = render_handwriting(
        "A1 b2.3",
        output_path=output_path,
        seed=1,
        target_height=80,
        digits_root=digits_root,
        letters_root=letters_root,
    )

    assert rendered_path == output_path
    assert rendered_path.exists()
    image = Image.open(rendered_path)
    assert image.width > 0
    assert image.height > 0


def test_render_handwriting_ignores_unsplit_source_images(tmp_path):
    digits_root = tmp_path / "font_digits"
    digit_dir = digits_root / "1"
    digit_dir.mkdir(parents=True)
    _create_character_sample(digit_dir / "digit_1_001.png", "1")
    _create_character_sample(digit_dir / "image.png", "11111")

    letters_root = tmp_path / "font_letters"
    letter_dir = letters_root / "uppercase" / "A"
    letter_dir.mkdir(parents=True)
    _create_character_sample(letter_dir / "letter_A_001.png", "A")
    _create_character_sample(letter_dir / "JPEG图像.jpeg", "AAAAA")

    output_path = tmp_path / "rendered_filtered.png"
    rendered_path = render_handwriting(
        "A1",
        output_path=output_path,
        seed=1,
        target_height=80,
        digits_root=digits_root,
        letters_root=letters_root,
    )

    assert rendered_path.exists()


def test_render_handwriting_supports_common_punctuation(tmp_path):
    letters_root = tmp_path / "font_letters"
    for case_folder, letter in [("uppercase", "H"), ("lowercase", "i")]:
        letter_dir = letters_root / case_folder / letter
        letter_dir.mkdir(parents=True)
        _create_character_sample(letter_dir / f"letter_{letter}_001.png", letter)

    output_path = tmp_path / "rendered_punctuation.png"
    rendered_path = render_handwriting(
        "Hi, Hi?!",
        output_path=output_path,
        seed=2,
        target_height=80,
        digits_root=tmp_path / "font_digits",
        letters_root=letters_root,
    )

    assert rendered_path.exists()
    image = Image.open(rendered_path)
    assert image.width > 0
    assert image.height > 0
