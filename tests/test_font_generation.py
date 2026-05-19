import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import font_generation
from font_generation import (
    TOP_PUNCTUATION_BOTTOM_TARGET,
    baseline_for_top_punctuation,
    build_font,
    find_sample_for,
    glyph_name_for,
)


def test_build_font_produces_valid_ttf(tmp_path):
    output = tmp_path / "test.ttf"
    build_font(output_path=output, family_name="TestFont")
    
    assert output.exists()
    assert output.stat().st_size > 0
    
    # Verify it's a valid font that fontTools can re-parse
    from fontTools.ttLib import TTFont
    font = TTFont(output)
    assert "A" in font.getGlyphOrder()
    assert "zero" in font.getGlyphOrder()
    assert font["name"].getDebugName(1) == "TestFont"


def test_find_sample_for_prefers_primary_font_samples(tmp_path, monkeypatch):
    samples_root = tmp_path / "samples"
    uppercase_dir = samples_root / "font_letters" / "uppercase" / "A"
    digit_dir = samples_root / "font_digits" / "7"
    punctuation_dir = samples_root / "font_symbols" / "period"
    uppercase_dir.mkdir(parents=True)
    digit_dir.mkdir(parents=True)
    punctuation_dir.mkdir(parents=True)

    (uppercase_dir / "letter_A_001.png").write_bytes(b"regular")
    primary_letter = uppercase_dir / "letter_A_000_primary.png"
    primary_letter.write_bytes(b"primary")

    (digit_dir / "digit_7_001.png").write_bytes(b"regular")
    primary_digit = digit_dir / "digit_7_000_primary.png"
    primary_digit.write_bytes(b"primary")

    (punctuation_dir / "symbol_period_001.png").write_bytes(b"regular")
    primary_period = punctuation_dir / "symbol_period_000_primary.png"
    primary_period.write_bytes(b"primary")

    monkeypatch.setattr(font_generation, "SAMPLES_ROOT", samples_root)

    assert find_sample_for("A") == primary_letter
    assert find_sample_for("7") == primary_digit
    assert find_sample_for(".") == primary_period


def test_glyph_name_for_supports_basic_punctuation():
    assert glyph_name_for(".") == "period"
    assert glyph_name_for(",") == "comma"
    assert glyph_name_for("?") == "question"
    assert glyph_name_for("!") == "exclam"
    assert glyph_name_for("'") == "quotesingle"
    assert glyph_name_for('"') == "quotedbl"
    assert glyph_name_for(":") == "colon"
    assert glyph_name_for(";") == "semicolon"
    assert glyph_name_for("-") == "hyphen"
    assert glyph_name_for("(") == "parenleft"
    assert glyph_name_for(")") == "parenright"


def test_top_punctuation_uses_raised_virtual_baseline():
    # white background with a simple quote-like vertical stroke near the top
    binary = __import__("numpy").ones((40, 20), dtype=bool)
    binary[4:20, 8:12] = False
    baseline = baseline_for_top_punctuation(binary, scale=1.0)
    assert baseline > 19
    assert baseline == 19 + TOP_PUNCTUATION_BOTTOM_TARGET
