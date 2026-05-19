from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from preprocessing import (
    _background_is_light,
    _crop_to_content,
    _load_grayscale,
    preprocess_digit_array_from_gray,
)

DEFAULT_ALPHABET_ROW_TEXT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n1234567890"
DEFAULT_BASIC_PUNCTUATION_ROW = ".,?!"
DEFAULT_EXTENDED_PUNCTUATION_ROW = ".,?!'\":;-()"
DEFAULT_ALPHABET_ROW_TEXT_WITH_PUNCTUATION = (
    f"{DEFAULT_ALPHABET_ROW_TEXT}\n{DEFAULT_BASIC_PUNCTUATION_ROW}"
)
DEFAULT_ALPHABET_ROW_TEXT_WITH_EXTENDED_PUNCTUATION = (
    f"{DEFAULT_ALPHABET_ROW_TEXT}\n{DEFAULT_EXTENDED_PUNCTUATION_ROW}"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_DIGITS_ROOT = PROJECT_ROOT / "samples" / "font_digits"
FONT_LETTERS_ROOT = PROJECT_ROOT / "samples" / "font_letters"
FONT_SYMBOLS_ROOT = PROJECT_ROOT / "samples" / "font_symbols"
PUNCTUATION_GLYPH_NAMES = {
    ".": "period",
    ",": "comma",
    "?": "question",
    "!": "exclam",
    "'": "quotesingle",
    '"': "quotedbl",
    ":": "colon",
    ";": "semicolon",
    "-": "hyphen",
    "(": "parenleft",
    ")": "parenright",
}


@dataclass(frozen=True)
class AlphabetRowSegment:
    index: int
    line_index: int
    column_index: int
    character: str | None
    bbox: tuple[int, int, int, int]
    raw_image_array: np.ndarray
    image_array: np.ndarray


@dataclass(frozen=True)
class _Box:
    bbox: tuple[int, int, int, int]
    area: float

    @property
    def x(self):
        return self.bbox[0]

    @property
    def y(self):
        return self.bbox[1]

    @property
    def width(self):
        return self.bbox[2]

    @property
    def height(self):
        return self.bbox[3]

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2


def default_expected_rows_for_line_count(line_count):
    if line_count == 3:
        return normalize_expected_rows(DEFAULT_ALPHABET_ROW_TEXT)
    if line_count == 4:
        return normalize_expected_rows(DEFAULT_ALPHABET_ROW_TEXT_WITH_PUNCTUATION)
    raise ValueError(
        f"Unsupported alphabet-row layout with {line_count} line(s). "
        "Expected either 3 lines (letters + digits) or 4 lines (letters + digits + punctuation)."
    )


def normalize_expected_rows(expected_text=None):
    source = expected_text or DEFAULT_ALPHABET_ROW_TEXT
    rows = []
    for line in source.splitlines():
        collapsed = "".join(line.split())
        if collapsed:
            rows.append(collapsed)
    return rows


def _infer_default_punctuation_row(symbol_count):
    if symbol_count == len(DEFAULT_BASIC_PUNCTUATION_ROW):
        return DEFAULT_BASIC_PUNCTUATION_ROW
    if symbol_count == len(DEFAULT_EXTENDED_PUNCTUATION_ROW):
        return DEFAULT_EXTENDED_PUNCTUATION_ROW
    raise ValueError(
        "Unsupported punctuation-row layout. "
        f"Detected {symbol_count} symbol(s), but currently only "
        f"{len(DEFAULT_BASIC_PUNCTUATION_ROW)}-symbol and "
        f"{len(DEFAULT_EXTENDED_PUNCTUATION_ROW)}-symbol punctuation rows are supported."
    )


def _merge_bbox(box_a, box_b):
    left = min(box_a[0], box_b[0])
    top = min(box_a[1], box_b[1])
    right = max(box_a[0] + box_a[2], box_b[0] + box_b[2])
    bottom = max(box_a[1] + box_a[3], box_b[1] + box_b[3])
    return left, top, right - left, bottom - top


def _padded_bbox(x, y, width, height, image_width, image_height):
    padding = max(3, int(round(max(width, height) * 0.14)))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return left, top, right - left, bottom - top


def _binarize_for_alphabet_row(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    threshold_type = cv2.THRESH_BINARY_INV if _background_is_light(gray_image) else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        41,
        8,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _find_boxes(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = binary_image.shape
    boxes = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 8 or width < 2 or height < 2:
            continue
        if width > image_width * 0.97 and height > image_height * 0.97:
            continue
        boxes.append((x, y, width, height, area))

    return boxes


def _estimate_character_height(raw_boxes):
    likely_heights = [height for _, _, width, height, area in raw_boxes if height >= 14 and width >= 3 and area >= 18]
    if not likely_heights:
        likely_heights = [height for _, _, _, height, _ in raw_boxes]
    if not likely_heights:
        return 0.0
    return float(np.median(likely_heights))


def _split_main_and_dot_boxes(raw_boxes, image_width, image_height):
    character_height = _estimate_character_height(raw_boxes)
    if character_height <= 0:
        return [], []

    main_boxes = []
    dot_boxes = []

    for x, y, width, height, area in raw_boxes:
        padded = _padded_bbox(x, y, width, height, image_width, image_height)
        box = _Box(padded, area)
        is_small_box = (
            height <= character_height * 0.36
            and width <= character_height * 0.42
            and area <= (character_height ** 2) * 0.1
        )
        if is_small_box:
            dot_boxes.append(box)
        elif height >= character_height * 0.34 or width >= character_height * 0.14:
            main_boxes.append(box)

    return main_boxes, dot_boxes


def _group_boxes_by_line(boxes):
    if not boxes:
        return []

    median_height = float(np.median([box.height for box in boxes]))
    line_tolerance = max(18, median_height * 0.8)
    lines = []

    for box in sorted(boxes, key=lambda item: item.center_y):
        matching_line = None
        for line in lines:
            line_center = np.mean([item.center_y for item in line])
            if abs(box.center_y - line_center) <= line_tolerance:
                matching_line = line
                break

        if matching_line is None:
            lines.append([box])
        else:
            matching_line.append(box)

    return sorted(lines, key=lambda line: np.mean([item.center_y for item in line]))


def _attach_dots_to_line(line_boxes, dot_boxes):
    if not dot_boxes:
        return sorted(line_boxes, key=lambda item: item.center_x), []

    merged = [box for box in line_boxes]
    used_indices = set()

    for index, dot_box in enumerate(dot_boxes):
        best_match = None
        best_score = None

        for line_index, line_box in enumerate(merged):
            horizontal_overlap = (
                dot_box.center_x >= line_box.x - line_box.width * 0.25
                and dot_box.center_x <= line_box.right + line_box.width * 0.25
            )
            vertical_gap = line_box.y - dot_box.bottom
            if not horizontal_overlap:
                continue
            if vertical_gap < -line_box.height * 0.15 or vertical_gap > line_box.height * 0.95:
                continue

            score = abs(dot_box.center_x - line_box.center_x) + max(0, vertical_gap) * 0.4
            if best_score is None or score < best_score:
                best_match = line_index
                best_score = score

        if best_match is not None:
            merged_box = _merge_bbox(merged[best_match].bbox, dot_box.bbox)
            merged[best_match] = _Box(merged_box, merged[best_match].area + dot_box.area)
            used_indices.add(index)

    remaining = [dot_box for index, dot_box in enumerate(dot_boxes) if index not in used_indices]
    return sorted(merged, key=lambda item: item.center_x), remaining


def _merge_fragmented_boxes(line_boxes):
    if not line_boxes:
        return []

    sorted_boxes = sorted(line_boxes, key=lambda item: item.x)
    merged = [sorted_boxes[0]]

    for current_box in sorted_boxes[1:]:
        previous_box = merged[-1]
        horizontal_overlap = min(previous_box.right, current_box.right) - max(previous_box.x, current_box.x)
        vertical_overlap = min(previous_box.bottom, current_box.bottom) - max(previous_box.y, current_box.y)
        gap = current_box.x - previous_box.right

        should_merge = (
            horizontal_overlap >= min(previous_box.width, current_box.width) * 0.3
            and vertical_overlap >= min(previous_box.height, current_box.height) * 0.45
        ) or (
            gap <= 1
            and vertical_overlap >= min(previous_box.height, current_box.height) * 0.65
        )

        if should_merge:
            merged_box = _merge_bbox(previous_box.bbox, current_box.bbox)
            merged[-1] = _Box(merged_box, previous_box.area + current_box.area)
        else:
            merged.append(current_box)

    return merged


def _merge_detached_accent_boxes(line_boxes):
    if not line_boxes:
        return []

    median_height = float(np.median([box.height for box in line_boxes]))
    boxes = sorted(line_boxes, key=lambda item: item.center_x)
    used_indices = set()
    merged = []

    for index, box in enumerate(boxes):
        if index in used_indices:
            continue

        is_small_top_box = box.height <= median_height * 0.3 and box.width <= median_height * 0.45
        if is_small_top_box:
            best_match = None
            best_score = None
            for candidate_index, candidate in enumerate(boxes):
                if candidate_index == index or candidate_index in used_indices:
                    continue
                if candidate.height <= median_height * 0.55:
                    continue

                horizontal_overlap = (
                    box.center_x >= candidate.x - candidate.width * 0.2
                    and box.center_x <= candidate.right + candidate.width * 0.2
                )
                vertical_position_ok = box.bottom <= candidate.y + candidate.height * 0.2
                if not horizontal_overlap or not vertical_position_ok:
                    continue

                score = abs(box.center_x - candidate.center_x) + abs(box.bottom - candidate.y) * 0.3
                if best_score is None or score < best_score:
                    best_match = candidate_index
                    best_score = score

            if best_match is not None:
                merged_box = _merge_bbox(box.bbox, boxes[best_match].bbox)
                used_indices.add(index)
                used_indices.add(best_match)
                merged.append(_Box(merged_box, box.area + boxes[best_match].area))
                continue

        merged.append(box)

    return sorted(merged, key=lambda item: item.center_x)


def _best_split_for_wide_box(box, binary_image):
    x, y, width, height = box.bbox
    if width < 60:
        return None

    crop = binary_image[y : y + height, x : x + width]
    if crop.size == 0:
        return None

    column_ink = np.count_nonzero(crop, axis=0)
    if column_ink.size < 10:
        return None

    start = max(3, int(width * 0.2))
    end = min(width - 3, int(width * 0.8))
    if end <= start:
        return None

    candidate_slice = column_ink[start:end]
    split_offset = int(np.argmin(candidate_slice)) + start
    if column_ink[split_offset] > column_ink.max() * 0.55:
        return None

    left_width = split_offset
    right_width = width - split_offset
    if left_width < 10 or right_width < 10:
        return None

    left_box = (x, y, left_width, height)
    right_box = (x + split_offset, y, right_width, height)
    return _Box(left_box, box.area / 2), _Box(right_box, box.area / 2)


def _split_wide_boxes_to_match_expected(line_boxes, binary_image, expected_count):
    boxes = [box for box in line_boxes]

    while len(boxes) < expected_count:
        widest = max(boxes, key=lambda item: item.width, default=None)
        if widest is None:
            break

        split_pair = _best_split_for_wide_box(widest, binary_image)
        if split_pair is None:
            break

        widest_index = boxes.index(widest)
        boxes = boxes[:widest_index] + [split_pair[0], split_pair[1]] + boxes[widest_index + 1 :]
        boxes = sorted(boxes, key=lambda item: item.center_x)

    return boxes


def _is_punctuation_only_row(expected_row):
    return bool(expected_row) and all(character in PUNCTUATION_GLYPH_NAMES for character in expected_row)


def _supplement_punctuation_row_with_standalone_marks(line_boxes, remaining_dots, expected_row):
    if not _is_punctuation_only_row(expected_row):
        return sorted(line_boxes, key=lambda item: item.center_x), remaining_dots
    if len(line_boxes) >= len(expected_row) or not remaining_dots:
        return sorted(line_boxes, key=lambda item: item.center_x), remaining_dots

    row_top = min(box.y for box in line_boxes)
    row_bottom = max(box.bottom for box in line_boxes)
    row_height = max(box.height for box in line_boxes)
    selected = []
    still_remaining = []

    for dot_box in remaining_dots:
        center_y = dot_box.center_y
        within_row_band = (
            center_y >= row_top - row_height * 0.2
            and center_y <= row_bottom + row_height * 0.8
        )
        if within_row_band:
            selected.append(dot_box)
        else:
            still_remaining.append(dot_box)

    supplemented = sorted(line_boxes + selected, key=lambda item: item.center_x)

    if len(supplemented) > len(expected_row):
        return supplemented[: len(expected_row)], still_remaining + supplemented[len(expected_row) :]

    return supplemented, still_remaining


def _merge_punctuation_row_symbol_boxes(candidate_boxes):
    if not candidate_boxes:
        return []

    sorted_boxes = sorted(candidate_boxes, key=lambda item: item.center_x)
    merged = [sorted_boxes[0]]

    for current_box in sorted_boxes[1:]:
        previous_box = merged[-1]
        same_symbol = abs(current_box.center_x - previous_box.center_x) <= max(
            18,
            min(previous_box.width, current_box.width) * 1.6,
        )
        if same_symbol:
            merged_box = _merge_bbox(previous_box.bbox, current_box.bbox)
            merged[-1] = _Box(merged_box, previous_box.area + current_box.area)
        else:
            merged.append(current_box)

    return merged


def _build_punctuation_row_boxes(line_boxes, remaining_dots, expected_row):
    if not _is_punctuation_only_row(expected_row):
        return sorted(line_boxes, key=lambda item: item.center_x), remaining_dots
    if not line_boxes and not remaining_dots:
        return [], []

    if line_boxes:
        row_top = min(box.y for box in line_boxes)
        row_bottom = max(box.bottom for box in line_boxes)
        row_height = max(box.height for box in line_boxes)
    else:
        row_top = min(box.y for box in remaining_dots)
        row_bottom = max(box.bottom for box in remaining_dots)
        row_height = max(box.height for box in remaining_dots)

    selected = []
    still_remaining = []
    for dot_box in remaining_dots:
        center_y = dot_box.center_y
        within_row_band = (
            center_y >= row_top - row_height * 0.2
            and center_y <= row_bottom + row_height * 1.2
        )
        if within_row_band:
            selected.append(dot_box)
        else:
            still_remaining.append(dot_box)

    merged_symbols = _merge_punctuation_row_symbol_boxes(line_boxes + selected)
    return merged_symbols, still_remaining


def segment_alphabet_rows(image_path, expected_text=None):
    image_path = Path(image_path)
    gray_image = _load_grayscale(image_path)
    binary_image = _binarize_for_alphabet_row(gray_image)
    raw_boxes = _find_boxes(binary_image)
    image_height, image_width = binary_image.shape
    main_boxes, dot_boxes = _split_main_and_dot_boxes(raw_boxes, image_width, image_height)
    line_boxes = _group_boxes_by_line(main_boxes)
    if not line_boxes:
        return []

    using_explicit_expected_text = expected_text is not None
    expected_rows = (
        normalize_expected_rows(expected_text)
        if using_explicit_expected_text
        else default_expected_rows_for_line_count(len(line_boxes))
    )

    if len(line_boxes) != len(expected_rows):
        raise ValueError(
            f"Expected {len(expected_rows)} row(s) from the known-content template, "
            f"but found {len(line_boxes)} line(s)."
        )

    segments_by_line = []
    next_index = 0
    remaining_dots = dot_boxes

    for line_index, boxes in enumerate(line_boxes):
        merged_boxes, remaining_dots = _attach_dots_to_line(boxes, remaining_dots)
        merged_boxes = _merge_fragmented_boxes(merged_boxes)
        merged_boxes = _merge_detached_accent_boxes(merged_boxes)
        expected_row = expected_rows[line_index]
        if not using_explicit_expected_text and line_index == len(line_boxes) - 1 and len(line_boxes) == 4:
            expected_row = None

        if expected_row is not None and _is_punctuation_only_row(expected_row):
            merged_boxes, remaining_dots = _build_punctuation_row_boxes(
                merged_boxes,
                remaining_dots,
                expected_row,
            )
        elif expected_row is not None:
            merged_boxes, remaining_dots = _supplement_punctuation_row_with_standalone_marks(
                merged_boxes,
                remaining_dots,
                expected_row,
            )
        else:
            merged_boxes, remaining_dots = _build_punctuation_row_boxes(
                merged_boxes,
                remaining_dots,
                DEFAULT_EXTENDED_PUNCTUATION_ROW,
            )
            expected_row = _infer_default_punctuation_row(len(merged_boxes))
        if len(merged_boxes) < len(expected_row):
            merged_boxes = _split_wide_boxes_to_match_expected(
                merged_boxes,
                binary_image,
                len(expected_row),
            )
        if len(merged_boxes) != len(expected_row):
            raise ValueError(
                f"Expected {len(expected_row)} character(s) in row {line_index + 1}, "
                f"but found {len(merged_boxes)}. The row image may need cleaner spacing."
            )

        # Compute the row's vertical extent once. All letters in this row will be
        # cropped to this same vertical span, preserving each letter's position
        # relative to the row's baseline. This means the per-letter PNGs retain
        # information about ascenders, descenders, and x-height, which the font
        # generator can then use to align glyphs correctly.
        row_top = min(box.y for box in merged_boxes)
        row_bottom = max(box.bottom for box in merged_boxes)

        line_segments = []
        for column_index, (box, character) in enumerate(zip(merged_boxes, expected_row)):
            x, _, width, _ = box.bbox
            raw_crop = gray_image[row_top:row_bottom, x : x + width]
            content_crop = _crop_to_content(_binarize_for_alphabet_row(raw_crop))
            if content_crop.size == 0:
                processed = preprocess_digit_array_from_gray(raw_crop)
            else:
                processed = preprocess_digit_array_from_gray(raw_crop)

            line_segments.append(
                AlphabetRowSegment(
                    index=next_index,
                    line_index=line_index,
                    column_index=column_index,
                    character=character,
                    bbox=box.bbox,
                    raw_image_array=raw_crop,
                    image_array=processed,
                )
            )
            next_index += 1

        segments_by_line.append(line_segments)

    return segments_by_line


def format_alphabet_row_prediction(rows):
    return "\n".join("".join(segment["text"] for segment in row) for row in rows if row)


def save_alphabet_row_segments(image_path, output_dir, expected_text=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for row_segments in segment_alphabet_rows(image_path, expected_text=expected_text):
        for segment in row_segments:
            character_label = segment.character or "unknown"
            output_path = output_dir / (
                f"segment_{segment.index + 1:03d}_{character_label}_line{segment.line_index + 1}.png"
            )
            Image.fromarray(segment.raw_image_array.astype(np.uint8)).save(output_path)
            saved_paths.append(output_path)

    return saved_paths


def _primary_font_sample_path(character):
    if character.isalpha():
        case = "uppercase" if character.isupper() else "lowercase"
        return FONT_LETTERS_ROOT / case / character / f"letter_{character}_000_primary.png"
    if character.isdigit():
        return FONT_DIGITS_ROOT / character / f"digit_{character}_000_primary.png"
    if character in PUNCTUATION_GLYPH_NAMES:
        glyph_name = PUNCTUATION_GLYPH_NAMES[character]
        return FONT_SYMBOLS_ROOT / glyph_name / f"symbol_{glyph_name}_000_primary.png"
    raise ValueError(f"Unsupported character for font library export: {character!r}")


def populate_font_library_from_alphabet_row(image_path, expected_text=None):
    saved_paths = []

    for row_segments in segment_alphabet_rows(image_path, expected_text=expected_text):
        for segment in row_segments:
            if segment.character is None:
                continue
            output_path = _primary_font_sample_path(segment.character)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(segment.raw_image_array.astype(np.uint8)).save(output_path)
            saved_paths.append(output_path)

    return saved_paths
