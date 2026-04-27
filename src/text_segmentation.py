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


@dataclass(frozen=True)
class TextSegment:
    index: int
    bbox: tuple[int, int, int, int]
    image_array: np.ndarray
    kind: str = "character"
    literal: str | None = None
    separator_before: bool = False


@dataclass(frozen=True)
class _RawBox:
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


@dataclass(frozen=True)
class _LineSymbol:
    bbox: tuple[int, int, int, int]
    kind: str
    literal: str | None = None

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


def _merge_bbox(box_a, box_b):
    left = min(box_a[0], box_b[0])
    top = min(box_a[1], box_b[1])
    right = max(box_a[0] + box_a[2], box_b[0] + box_b[2])
    bottom = max(box_a[1] + box_a[3], box_b[1] + box_b[3])
    return left, top, right - left, bottom - top


def _padded_bbox(x, y, width, height, image_width, image_height):
    padding = max(3, int(round(max(width, height) * 0.16)))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return left, top, right - left, bottom - top


def _binarize_for_text(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    threshold_type = cv2.THRESH_BINARY_INV if _background_is_light(gray_image) else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        41,
        9,
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, close_kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)


def _find_raw_boxes(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = binary_image.shape
    boxes = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 8 or width < 2 or height < 2:
            continue
        if width > image_width * 0.95 and height > image_height * 0.95:
            continue
        boxes.append((x, y, width, height, area))

    return boxes


def _estimate_character_height(raw_boxes):
    likely_heights = [height for _, _, width, height, area in raw_boxes if height >= 12 and width >= 3 and area >= 18]
    if not likely_heights:
        likely_heights = [height for _, _, _, height, _ in raw_boxes]
    if not likely_heights:
        return 0.0
    return float(np.median(likely_heights))


def _classify_boxes(raw_boxes, image_width, image_height):
    character_height = _estimate_character_height(raw_boxes)
    if character_height <= 0:
        return [], []

    main_boxes = []
    small_boxes = []

    for x, y, width, height, area in raw_boxes:
        padded = _padded_bbox(x, y, width, height, image_width, image_height)
        if height >= character_height * 0.38 or width >= character_height * 0.16:
            main_boxes.append(_RawBox(padded, area))
        elif height <= character_height * 0.34 and width <= character_height * 0.34:
            small_boxes.append(_RawBox(padded, area))

    return main_boxes, small_boxes


def _group_boxes_by_line(boxes):
    if not boxes:
        return []

    median_height = float(np.median([box.height for box in boxes]))
    line_tolerance = max(16, median_height * 0.8)
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


def _attach_diacritics(line_boxes, small_boxes):
    used_indices = set()
    merged_boxes = [box for box in line_boxes]

    for index, dot in enumerate(small_boxes):
        best_match = None
        best_score = None

        for line_index, line_box in enumerate(merged_boxes):
            horizontal_overlap = (
                dot.center_x >= line_box.x - line_box.width * 0.25
                and dot.center_x <= line_box.right + line_box.width * 0.25
            )
            vertical_gap = line_box.y - dot.bottom
            if not horizontal_overlap:
                continue
            if vertical_gap < -line_box.height * 0.15 or vertical_gap > line_box.height * 0.95:
                continue

            score = abs(dot.center_x - line_box.center_x) + max(0, vertical_gap) * 0.4
            if best_score is None or score < best_score:
                best_match = line_index
                best_score = score

        if best_match is not None:
            merged_box = _merge_bbox(merged_boxes[best_match].bbox, dot.bbox)
            merged_boxes[best_match] = _RawBox(merged_box, merged_boxes[best_match].area + dot.area)
            used_indices.add(index)

    remaining_dots = [dot for index, dot in enumerate(small_boxes) if index not in used_indices]
    return merged_boxes, remaining_dots


def _merge_detached_accents(main_boxes):
    if not main_boxes:
        return []

    sorted_boxes = sorted(main_boxes, key=lambda item: item.x)
    median_height = float(np.median([box.height for box in sorted_boxes]))
    merged_boxes = []
    index = 0

    while index < len(sorted_boxes):
        current_box = sorted_boxes[index]
        if index + 1 < len(sorted_boxes):
            next_box = sorted_boxes[index + 1]
            is_small_top_box = current_box.height <= median_height * 0.32
            overlaps_next = (
                current_box.center_x >= next_box.x - next_box.width * 0.2
                and current_box.center_x <= next_box.right + next_box.width * 0.2
            )
            sits_above_next = current_box.bottom <= next_box.y + next_box.height * 0.35
            if is_small_top_box and overlaps_next and sits_above_next:
                merged_boxes.append(_RawBox(_merge_bbox(current_box.bbox, next_box.bbox), current_box.area + next_box.area))
                index += 2
                continue

        merged_boxes.append(current_box)
        index += 1

    return merged_boxes


def _dot_belongs_to_line(dot_box, line_boxes):
    min_y = min(box.y for box in line_boxes)
    max_y = max(box.bottom for box in line_boxes)
    median_height = float(np.median([box.height for box in line_boxes]))
    return min_y - median_height * 0.25 <= dot_box.center_y <= max_y + median_height * 0.45


def _best_vertical_split(box, binary_image, median_width):
    x, y, width, height = box.bbox
    if width < max(median_width * 1.15, 78):
        return None

    crop = binary_image[y : y + height, x : x + width]
    if crop.size == 0:
        return None

    column_sums = crop.sum(axis=0).astype(np.float32) / 255.0
    if len(column_sums) < 12:
        return None

    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(column_sums, kernel, mode="same")

    left_margin = max(3, int(round(width * 0.18)))
    right_margin = width - left_margin
    if right_margin <= left_margin + 2:
        return None

    candidate_values = smoothed[left_margin:right_margin]
    if candidate_values.size == 0:
        return None

    split_column = int(np.argmin(candidate_values)) + left_margin
    valley_value = float(smoothed[split_column])
    max_value = float(smoothed.max()) if smoothed.size else 0.0
    if max_value <= 0:
        return None

    left_peak = float(smoothed[:split_column].max()) if split_column > 0 else 0.0
    right_peak = float(smoothed[split_column + 1 :].max()) if split_column + 1 < len(smoothed) else 0.0
    if left_peak <= 0 or right_peak <= 0:
        return None

    left_width = split_column
    right_width = width - split_column
    min_side_width = max(10, int(round(median_width * 0.28)))
    if left_width < min_side_width or right_width < min_side_width:
        return None

    if valley_value > max_value * 0.26:
        return None
    if valley_value > min(left_peak, right_peak) * 0.32:
        return None

    return split_column


def _split_wide_boxes(main_boxes, binary_image):
    if not main_boxes:
        return []

    median_width = float(np.median([box.width for box in main_boxes]))
    split_boxes = []

    for box in sorted(main_boxes, key=lambda item: item.x):
        split_column = _best_vertical_split(box, binary_image, median_width)
        if split_column is None:
            split_boxes.append(box)
            continue

        x, y, width, height = box.bbox
        left_box = _RawBox((x, y, split_column, height), box.area / 2)
        right_box = _RawBox((x + split_column, y, width - split_column, height), box.area / 2)
        split_boxes.extend([left_box, right_box])

    return split_boxes


def _line_symbols(main_boxes, dot_boxes, binary_image):
    merged_main_boxes, remaining_dots = _attach_diacritics(main_boxes, dot_boxes)
    merged_main_boxes = _merge_detached_accents(merged_main_boxes)
    merged_main_boxes = _split_wide_boxes(merged_main_boxes, binary_image)
    merged_main_boxes = sorted(merged_main_boxes, key=lambda item: item.x)

    punctuation = []
    for dot in remaining_dots:
        if _dot_belongs_to_line(dot, merged_main_boxes):
            punctuation.append(_LineSymbol(dot.bbox, kind="punctuation", literal="."))

    symbols = [_LineSymbol(box.bbox, kind="character") for box in merged_main_boxes]
    symbols.extend(punctuation)
    return sorted(symbols, key=lambda item: item.center_x)


def _space_threshold(main_boxes):
    if len(main_boxes) < 2:
        return 18

    widths = [box.width for box in main_boxes]
    gaps = []
    for previous_box, next_box in zip(main_boxes, main_boxes[1:]):
        gap = next_box.x - previous_box.right
        if gap > 0:
            gaps.append(gap)

    median_width = float(np.median(widths)) if widths else 22.0
    median_gap = float(np.median(gaps)) if gaps else median_width * 0.25
    return max(median_width * 0.3, median_gap * 0.55, 14)


def segment_text_lines(image_path):
    image_path = Path(image_path)
    gray_image = _load_grayscale(image_path)
    binary_image = _binarize_for_text(gray_image)
    image_height, image_width = binary_image.shape

    raw_boxes = _find_raw_boxes(binary_image)
    main_boxes, small_boxes = _classify_boxes(raw_boxes, image_width, image_height)
    main_lines = _group_boxes_by_line(main_boxes)

    if not main_lines:
        return []

    dot_lines = []
    for line_boxes in main_lines:
        line_dot_boxes = [dot for dot in small_boxes if _dot_belongs_to_line(dot, line_boxes)]
        dot_lines.append(line_dot_boxes)

    segment_lines = []
    current_index = 0

    for line_boxes, dot_boxes in zip(main_lines, dot_lines):
        symbols = _line_symbols(line_boxes, dot_boxes, binary_image)
        threshold = _space_threshold(sorted(line_boxes, key=lambda item: item.x))
        line_segments = []
        previous_symbol = None

        for symbol in symbols:
            x, y, width, height = symbol.bbox
            gray_crop = gray_image[y : y + height, x : x + width]
            centered = preprocess_digit_array_from_gray(gray_crop)

            separator_before = False
            if previous_symbol is not None and symbol.kind == "character":
                gap = symbol.x - previous_symbol.right
                separator_before = gap > threshold

            line_segments.append(
                TextSegment(
                    index=current_index,
                    bbox=symbol.bbox,
                    image_array=centered,
                    kind=symbol.kind,
                    literal=symbol.literal,
                    separator_before=separator_before,
                )
            )
            current_index += 1
            previous_symbol = symbol

        segment_lines.append(line_segments)

    return segment_lines


def save_text_segments(image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for line_index, line_segments in enumerate(segment_text_lines(image_path), start=1):
        for segment in line_segments:
            suffix = "char" if segment.kind == "character" else "punct"
            output_path = output_dir / f"line_{line_index:02}_segment_{segment.index + 1:03}_{suffix}.png"
            Image.fromarray(segment.image_array).save(output_path)
            saved_paths.append(output_path)

    return saved_paths
