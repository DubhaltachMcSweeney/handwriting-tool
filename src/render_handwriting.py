import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_DIGITS_DIR = PROJECT_ROOT / "samples" / "font_digits"
FONT_LETTERS_DIR = PROJECT_ROOT / "samples" / "font_letters"
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_HEIGHT = 140
DEFAULT_CHAR_SPACING_RATIO = 0.12
DEFAULT_SPACE_WIDTH_RATIO = 0.35
DEFAULT_LINE_SPACING_RATIO = 0.35
DEFAULT_MARGIN_RATIO = 0.18
DEFAULT_BASELINE_JITTER_RATIO = 0.06
SUPPORTED_PUNCTUATION = {".", ",", "!", "?", "'", ":", ";", "-"}


def _crop_white_margin(image):
    image = image.convert("RGBA")
    rgba = np.array(image)

    non_white_mask = np.any(rgba[:, :, :3] < 245, axis=2)
    if rgba.shape[2] == 4:
        non_white_mask &= rgba[:, :, 3] > 0

    coordinates = np.argwhere(non_white_mask)
    if coordinates.size == 0:
        return image

    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0) + 1
    top, left = top_left
    bottom, right = bottom_right
    return image.crop((left, top, right, bottom))


def _resize_preserve_aspect(image, target_height):
    width, height = image.size
    if height == 0:
        return image
    scale = target_height / height
    new_width = max(1, int(round(width * scale)))
    return image.resize((new_width, target_height), Image.LANCZOS)


def _character_variants(character, digits_root=FONT_DIGITS_DIR, letters_root=FONT_LETTERS_DIR):
    if character.isdigit():
        char_dir = Path(digits_root) / character
        label = f"digit {character}"
        pattern = "digit_*.png"
    elif character.isalpha():
        case_folder = "uppercase" if character.isupper() else "lowercase"
        char_dir = Path(letters_root) / case_folder / character
        label = f"letter {character}"
        pattern = "letter_*.png"
    else:
        raise ValueError(f"Unsupported character library lookup for: {character!r}")

    if not char_dir.exists():
        raise FileNotFoundError(f"Library folder not found for {label}: {char_dir}")

    variants = sorted(path for path in char_dir.glob(pattern) if path.is_file())
    if not variants:
        raise FileNotFoundError(f"No character images found in: {char_dir}")

    return variants


def load_character_glyph(
    character,
    rng,
    target_height=DEFAULT_HEIGHT,
    digits_root=FONT_DIGITS_DIR,
    letters_root=FONT_LETTERS_DIR,
):
    variant_path = rng.choice(
        _character_variants(character, digits_root=digits_root, letters_root=letters_root)
    )
    image = Image.open(variant_path)
    image = _crop_white_margin(image)
    return _resize_preserve_aspect(image, target_height)


def create_dot_glyph(target_height=DEFAULT_HEIGHT):
    dot_size = max(8, int(round(target_height * 0.16)))
    canvas_height = target_height
    canvas_width = max(dot_size + 6, int(round(target_height * 0.20)))
    image = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))

    draw = ImageDraw.Draw(image)
    x0 = (canvas_width - dot_size) // 2
    y0 = canvas_height - dot_size - max(4, dot_size // 3)
    draw.ellipse((x0, y0, x0 + dot_size, y0 + dot_size), fill=(0, 0, 0, 255))
    return image


def create_punctuation_glyph(character, target_height=DEFAULT_HEIGHT):
    if character == ".":
        return create_dot_glyph(target_height)

    stroke_width = max(3, int(round(target_height * 0.06)))
    dot_size = max(8, int(round(target_height * 0.15)))
    canvas_height = target_height
    canvas_width = max(dot_size + stroke_width + 8, int(round(target_height * 0.24)))
    image = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    center_x = canvas_width // 2
    lower_dot_top = canvas_height - dot_size - max(4, dot_size // 3)
    upper_dot_top = max(8, int(round(target_height * 0.30)))

    if character == ",":
        draw.ellipse(
            (center_x - dot_size // 2, lower_dot_top, center_x + dot_size // 2, lower_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
        draw.line(
            [
                (center_x + dot_size // 6, lower_dot_top + dot_size - 2),
                (center_x - dot_size // 3, lower_dot_top + dot_size + max(8, stroke_width * 2)),
            ],
            fill=(0, 0, 0, 255),
            width=stroke_width,
        )
    elif character == "!":
        draw.line(
            [
                (center_x, max(6, int(round(target_height * 0.10)))),
                (center_x, canvas_height - dot_size - max(16, stroke_width * 3)),
            ],
            fill=(0, 0, 0, 255),
            width=stroke_width,
        )
        draw.ellipse(
            (center_x - dot_size // 2, lower_dot_top, center_x + dot_size // 2, lower_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
    elif character == "?":
        arc_box = (
            center_x - int(round(target_height * 0.18)),
            max(4, int(round(target_height * 0.06))),
            center_x + int(round(target_height * 0.18)),
            int(round(target_height * 0.40)),
        )
        draw.arc(arc_box, start=200, end=20, fill=(0, 0, 0, 255), width=stroke_width)
        draw.line(
            [
                (center_x + int(round(target_height * 0.05)), int(round(target_height * 0.33))),
                (center_x, int(round(target_height * 0.55))),
            ],
            fill=(0, 0, 0, 255),
            width=stroke_width,
        )
        draw.ellipse(
            (center_x - dot_size // 2, lower_dot_top, center_x + dot_size // 2, lower_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
    elif character == "'":
        draw.line(
            [
                (center_x + max(1, stroke_width // 3), upper_dot_top),
                (center_x - max(2, stroke_width // 2), upper_dot_top + max(12, stroke_width * 3)),
            ],
            fill=(0, 0, 0, 255),
            width=stroke_width,
        )
    elif character == ":":
        draw.ellipse(
            (center_x - dot_size // 2, upper_dot_top, center_x + dot_size // 2, upper_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
        draw.ellipse(
            (center_x - dot_size // 2, lower_dot_top, center_x + dot_size // 2, lower_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
    elif character == ";":
        draw.ellipse(
            (center_x - dot_size // 2, upper_dot_top, center_x + dot_size // 2, upper_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
        draw.ellipse(
            (center_x - dot_size // 2, lower_dot_top, center_x + dot_size // 2, lower_dot_top + dot_size),
            fill=(0, 0, 0, 255),
        )
        draw.line(
            [
                (center_x + dot_size // 6, lower_dot_top + dot_size - 2),
                (center_x - dot_size // 3, lower_dot_top + dot_size + max(8, stroke_width * 2)),
            ],
            fill=(0, 0, 0, 255),
            width=stroke_width,
        )
    elif character == "-":
        draw.line(
            [
                (max(4, center_x - int(round(target_height * 0.12))), canvas_height // 2),
                (min(canvas_width - 4, center_x + int(round(target_height * 0.12))), canvas_height // 2),
            ],
            fill=(0, 0, 0, 255),
            width=stroke_width,
        )
    else:
        raise ValueError(
            f"Unsupported punctuation: {character!r}. "
            f"Supported punctuation is: {''.join(sorted(SUPPORTED_PUNCTUATION))}"
        )

    return image


def build_line_glyphs(
    text,
    rng,
    target_height=DEFAULT_HEIGHT,
    digits_root=FONT_DIGITS_DIR,
    letters_root=FONT_LETTERS_DIR,
):
    glyphs = []
    for char in text:
        if char.isdigit() or char.isalpha():
            glyphs.append(
                (
                    "glyph",
                    load_character_glyph(
                        char,
                        rng,
                        target_height,
                        digits_root=digits_root,
                        letters_root=letters_root,
                    ),
                )
            )
        elif char in SUPPORTED_PUNCTUATION:
            glyphs.append(("glyph", create_punctuation_glyph(char, target_height)))
        elif char == " ":
            glyphs.append(("space", None))
        else:
            raise ValueError(
                f"Unsupported character: {char!r}. "
                "This renderer currently supports digits, letters, spaces, common punctuation, and newlines."
            )
    return glyphs


def render_handwriting(
    text,
    output_path=None,
    seed=None,
    target_height=DEFAULT_HEIGHT,
    digits_root=FONT_DIGITS_DIR,
    letters_root=FONT_LETTERS_DIR,
):
    if not text:
        raise ValueError("Input text must not be empty.")

    rng = random.Random(seed)
    lines = text.splitlines() or [text]

    char_spacing = max(4, int(round(target_height * DEFAULT_CHAR_SPACING_RATIO)))
    space_width = max(12, int(round(target_height * DEFAULT_SPACE_WIDTH_RATIO)))
    line_spacing = max(10, int(round(target_height * DEFAULT_LINE_SPACING_RATIO)))
    margin = max(10, int(round(target_height * DEFAULT_MARGIN_RATIO)))
    baseline_jitter = max(2, int(round(target_height * DEFAULT_BASELINE_JITTER_RATIO)))

    line_layouts = []
    max_line_width = 0
    total_height = margin * 2

    for line in lines:
        glyph_specs = build_line_glyphs(
            line,
            rng,
            target_height=target_height,
            digits_root=digits_root,
            letters_root=letters_root,
        )
        rendered_items = []
        line_width = 0
        line_height = target_height + baseline_jitter * 2

        for kind, glyph in glyph_specs:
            if kind == "space":
                rendered_items.append((kind, space_width, None))
                line_width += space_width
            else:
                jitter = rng.randint(-baseline_jitter, baseline_jitter)
                rendered_items.append((kind, glyph.width, glyph, jitter))
                line_width += glyph.width

        if rendered_items:
            line_width += char_spacing * (len(rendered_items) - 1)

        line_layouts.append((rendered_items, line_width, line_height))
        max_line_width = max(max_line_width, line_width)
        total_height += line_height

    total_height += line_spacing * max(0, len(line_layouts) - 1)
    canvas_width = max(max_line_width + margin * 2, margin * 2 + 1)
    canvas = Image.new("RGBA", (canvas_width, total_height), (255, 255, 255, 255))

    current_y = margin
    for rendered_items, line_width, line_height in line_layouts:
        current_x = margin
        baseline_y = current_y + target_height + baseline_jitter
        for item in rendered_items:
            kind = item[0]
            width = item[1]
            if kind == "glyph":
                glyph = item[2]
                jitter = item[3]
                if glyph is not None:
                    y_offset = baseline_y - glyph.height + jitter
                    canvas.alpha_composite(glyph, dest=(current_x, y_offset))
            current_x += width + char_spacing
        current_y += line_height + line_spacing

    output_path = Path(output_path) if output_path else default_output_path(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    return output_path


def default_output_path(text):
    safe_name = re.sub(r"[^0-9A-Za-z.,!?':;\-\n ]+", "", text).replace("\n", "_").replace(" ", "-")
    safe_name = safe_name or "rendered"
    safe_name = safe_name[:60]
    return OUTPUT_DIR / f"rendered_{safe_name}.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render text using the handwritten digit and letter libraries."
    )
    parser.add_argument(
        "text",
        help="Text to render. Supports digits, letters, spaces, common punctuation, and newlines.",
    )
    parser.add_argument("--output", help="Output image path.")
    parser.add_argument("--seed", type=int, help="Optional random seed for deterministic variant selection.")
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Target character height in pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = render_handwriting(
        args.text,
        output_path=args.output,
        seed=args.seed,
        target_height=args.height,
    )
    print(f"Rendered handwriting saved to {output_path}")


if __name__ == "__main__":
    main()
