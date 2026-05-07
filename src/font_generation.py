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
ASCENT = 900
DESCENT = -200
ADVANCE_MARGIN = 50      # extra space added to each side of glyph for spacing

CAP_HEIGHT = 700  # target height for uppercase letters in font units

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

def find_sample_for(character):
    """Return path to the first available sample PNG for a character, or None."""
    if character.isalpha():
        case = "uppercase" if character.isupper() else "lowercase"
        directory = SAMPLES_ROOT / "font_letters" / case / character
        primary = directory / f"letter_{character}_000_primary.png"
        if primary.exists():
            return primary
        candidates = sorted(directory.glob(f"letter_{character}_*.png"))
    elif character.isdigit():
        directory = SAMPLES_ROOT / "font_digits" / character
        primary = directory / f"digit_{character}_000_primary.png"
        if primary.exists():
            return primary
        candidates = sorted(directory.glob(f"digit_{character}_*.png"))
    else:
        return None

    return candidates[0] if candidates else None

DESCENDER_LETTERS = set("gjpqy")

def detect_baseline_for_glyph(binary_array, character):
    """Find where the baseline should be for a single glyph, in PNG-Y coordinates.
    
    For non-descender characters, baseline = bottom of ink.
    For descender characters, baseline = where the body ends (roughly the top of the descender).
    """
    ink_mask = ~binary_array
    if not ink_mask.any():
        return 0
    
    rows = np.any(ink_mask, axis=1)
    ink_top = int(np.where(rows)[0][0])
    ink_bottom = int(np.where(rows)[0][-1])
    
    if character not in DESCENDER_LETTERS:
        return ink_bottom
    
    # j is a special case — there's no clear body-vs-descender density transition,
    # so use a fixed ratio instead of density-based detection.
    if character == 'j':
        # j's body sits roughly at 50% of total ink height
        return ink_top + int((ink_bottom - ink_top) * 0.5)

    # For other descenders (g, p, q, y), use density-based detection
    ink_per_row = ink_mask.sum(axis=1)
    
    # The body is in the upper portion. Find the row where ink density drops sharply.
    # Take the median ink-per-row in the upper half as "body density".
    upper_half_median = np.median(ink_per_row[ink_top:ink_top + (ink_bottom - ink_top) // 2 + 1])
    threshold = upper_half_median * 0.75  # consider rows with <75% of body density as descender
    
    # Walk down from the top, find the first row below the body that's "thin" (descender)
    body_end = ink_bottom  # fallback
    for y in range(ink_top + (ink_bottom - ink_top) // 2, ink_bottom):
        if ink_per_row[y] < threshold:
            body_end = y
            break
    
    return body_end

def trace_to_glyph(binary_array, scale, baseline_y_in_png):
    """Trace a binary array with Potrace and convert to a fontTools glyph.
    
    Coordinate transforms:
    - Y-axis flip (image Y down -> font Y up)
    - Translate so baseline_y_in_png maps to font Y=0
    - Scale by the provided factor
    """
    bitmap = potrace.Bitmap(binary_array)
    path = bitmap.trace(
        turdsize=2,        # smaller = preserves smaller features
        alphamax=1.0,      # smaller = more corners (less smoothing)
        opttolerance=0.1,  # smaller = closer to original
    )

    pen = TTGlyphPen(None)
    image_height, image_width = binary_array.shape

    def transform(x, y):
        return x * scale, (baseline_y_in_png - y) * scale

    for curve in path:
        start = curve.start_point
        pen.moveTo(transform(start.x, start.y))
        for segment in curve:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                pen.lineTo(transform(c.x, c.y))
                pen.lineTo(transform(end.x, end.y))
            else:
                c1, c2 = segment.c1, segment.c2
                end = segment.end_point
                pen.qCurveTo(
                    transform(c1.x, c1.y),
                    transform(c2.x, c2.y),
                    transform(end.x, end.y),
                )
        pen.closePath()

    return pen.glyph(), int(image_width * scale)

def compute_font_scale(reference_character="H"):
    """Compute a single scale factor for the whole font based on the actual ink
    height of a reference uppercase glyph (rather than the PNG canvas height,
    which now includes whitespace above and below for baseline preservation).
    """
    sample_path = find_sample_for(reference_character)
    if sample_path is None:
        raise FileNotFoundError(
            f"Cannot compute font scale: reference letter {reference_character!r} not found in samples."
        )
    binary = load_binary_image(sample_path)
    
    # Find the actual ink bounds (not the PNG canvas)
    ink_mask = ~binary
    if not ink_mask.any():
        raise ValueError(f"Reference glyph {reference_character!r} contains no ink")
    rows = np.any(ink_mask, axis=1)
    ink_top, ink_bottom = np.where(rows)[0][[0, -1]]
    ink_height = ink_bottom - ink_top + 1
    
    return CAP_HEIGHT / ink_height

def build_font(output_path=OUTPUT_TTF, family_name="MyHandwriting"):
    """Build the full TTF from all available character samples in SAMPLES_ROOT."""
    characters = (
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
        list("abcdefghijklmnopqrstuvwxyz") +
        list("0123456789")
    )

    # Compute a single scale factor from a reference uppercase glyph.
    # This preserves relative heights between ascenders, x-height, and descenders.
    scale = compute_font_scale(reference_character="H")

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
        baseline_y = detect_baseline_for_glyph(binary, character)
        glyph, width = trace_to_glyph(binary, scale, baseline_y_in_png=baseline_y)
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
