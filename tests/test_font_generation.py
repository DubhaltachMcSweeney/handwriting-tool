import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from font_generation import build_font


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