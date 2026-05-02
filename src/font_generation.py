import argparse
from pathlib import Path

import numpy as np
import potrace
from PIL import Image
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = PROJECT_ROOT / "samples"
OUTPUT_TTF = PROJECT_ROOT / "output" / "MyHandwriting.ttf"

UNITS_PER_EM = 1000
GLYPH_HEIGHT = 700        # cap-height target in font units
ASCENT = 800
DESCENT = -200
ADVANCE_MARGIN = 100      # extra space added to each side of glyph for spacing

def glyph_name_for(character):
    """Return a TTF-safe glyph name for a character.

    Uppercase letters: 'A' .. 'Z'
    Lowercase letters: 'a' .. 'z'   (lowercase glyph names are valid)
    Digits: 'zero', 'one', ... 'nine' (TTF prefers names, not digits)
    """
    if character.isupper() and character.isalpha():
        return character
    if character.islower() and character.isalpha():
        return character
    digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    if character.isdigit():
        return digit_names[int(character)]
    raise ValueError(f"Unsupported character: {character!r}")


def load_binary_image(png_path):
    """Load PNG and produce a boolean array suitable for potrace.

    potrace.Bitmap calls self.invert() on init, so we pass the inverse:
    True = background, False = ink. After potrace inverts, ink becomes the
    traced foreground.
    """
    image = Image.open(png_path).convert("L")
    array = np.array(image)
    return array >= 128


def trace_to_glyph(binary_array):
    """Trace a binary array with Potrace and convert to a fontTools glyph.

    Coordinate transforms:
    - Y-axis flip (image Y down -> font Y up)
    - Scale to GLYPH_HEIGHT in font units
    """
    bitmap = potrace.Bitmap(binary_array)
    path = bitmap.trace()

    pen = TTGlyphPen(None)
    image_height, image_width = binary_array.shape
    scale = GLYPH_HEIGHT / image_height

    for curve in path:
        start = curve.start_point
        pen.moveTo((start.x * scale, (image_height - start.y) * scale))
        for segment in curve:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                pen.lineTo((c.x * scale, (image_height - c.y) * scale))
                pen.lineTo((end.x * scale, (image_height - end.y) * scale))
            else:
                c1, c2 = segment.c1, segment.c2
                end = segment.end_point
                pen.qCurveTo(
                    (c1.x * scale, (image_height - c1.y) * scale),
                    (c2.x * scale, (image_height - c2.y) * scale),
                    (end.x * scale, (image_height - end.y) * scale),
                )
        pen.closePath()

    return pen.glyph(), int(image_width * scale)

def find_sample_for(character):
    """Return path to the first available sample PNG for a character, or None."""
    if character.isalpha():
        case = "uppercase" if character.isupper() else "lowercase"
        directory = SAMPLES_ROOT / "font_letters" / case / character
        candidates = sorted(directory.glob(f"letter_{character}_*.png"))
    elif character.isdigit():
        directory = SAMPLES_ROOT / "font_digits" / character
        candidates = sorted(directory.glob(f"digit_{character}_*.png"))
    else:
        return None

    return candidates[0] if candidates else None

def build_font(output_path=OUTPUT_TTF, family_name="MyHandwriting"):
    """Build the full TTF from all available character samples in SAMPLES_ROOT."""
    characters = (
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
        list("abcdefghijklmnopqrstuvwxyz") +
        list("0123456789")
    )

    glyphs = {}
    advance_widths = {}
    cmap = {}
    glyph_order = [".notdef"]

    # Empty .notdef - this is required by every TTF, shown when a character has no glyph.
    glyphs[".notdef"] = TTGlyphPen(None).glyph()
    advance_widths[".notdef"] = 500

    # Space — a non-printing glyph users will type for word breaks.
    glyphs["space"] = TTGlyphPen(None).glyph()
    advance_widths["space"] = 300
    cmap[0x20] = "space"
    glyph_order.append("space")

    skipped = []
    for character in characters:
        sample_path = find_sample_for(character)
        if sample_path is None:
            skipped.append(character)
            continue

        binary = load_binary_image(sample_path)
        glyph, width = trace_to_glyph(binary)
        name = glyph_name_for(character)

        glyphs[name] = glyph
        advance_widths[name] = width + ADVANCE_MARGIN
        cmap[ord(character)] = name
        glyph_order.append(name)

    # Assemble the TTF.
    fb = FontBuilder(UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({name: (advance_widths[name], 50) for name in glyph_order})
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({
        "familyName": family_name,
        "styleName": "Regular",
        "uniqueFontIdentifier": f"{family_name}-Regular-1.0",
        "fullName": f"{family_name} Regular",
        "psName": f"{family_name}-Regular",
        "version": "Version 1.0",
    })
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT, usWinAscent=ASCENT, usWinDescent=-DESCENT)
    fb.setupPost()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fb.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a TTF from handwriting samples.")
    parser.add_argument("--output", type=Path, default=OUTPUT_TTF, help="Output TTF path")
    parser.add_argument("--family-name", default="MyHandwriting", help="Font family name")
    args = parser.parse_args()
    
    build_font(output_path=args.output, family_name=args.family_name)
    print(f"Saved {args.output}")
