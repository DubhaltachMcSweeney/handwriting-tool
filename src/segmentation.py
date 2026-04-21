from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from preprocessing import (
    _background_is_light,
    _load_grayscale,
    _resize_and_center,
    digit_array_to_tensor,
)


@dataclass(frozen=True)
class DigitSegment:
    index: int
    bbox: tuple[int, int, int, int]
    image_array: np.ndarray
    kind: str = "digit"


@dataclass(frozen=True)
class _SymbolBox:
    bbox: tuple[int, int, int, int]
    area: float
    kind: str

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
    def center_y(self):
        return self.y + self.height / 2


def _padded_bbox(x, y, width, height, image_width, image_height):
    padding = max(2, int(round(max(width, height) * 0.15)))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return left, top, right - left, bottom - top


def _find_raw_boxes(binary_image):
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_height, image_width = binary_image.shape
    boxes = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 10 or width < 3 or height < 3:
            continue
        if width > image_width * 0.8 and height > image_height * 0.8:
            continue
        boxes.append((x, y, width, height, area))

    return boxes


def _estimate_digit_height(raw_boxes, image_height):
    likely_digit_heights = [
        height
        for _, _, width, height, area in raw_boxes
        if height >= image_height * 0.04 and width >= 6 and area >= 30
    ]
    if not likely_digit_heights:
        likely_digit_heights = [height for _, _, _, height, _ in raw_boxes]
    if not likely_digit_heights:
        return 0
    return float(np.median(likely_digit_heights))


def _classify_boxes(raw_boxes, image_width, image_height):
    digit_height = _estimate_digit_height(raw_boxes, image_height)
    if digit_height <= 0:
        return []

    symbols = []
    for x, y, width, height, area in raw_boxes:
        padded_bbox = _padded_bbox(x, y, width, height, image_width, image_height)
        is_digit = height >= digit_height * 0.45 and width >= 4
        is_dot = (
            height <= digit_height * 0.35
            and width <= digit_height * 0.35
            and area >= 12
        )

        if is_digit:
            symbols.append(_SymbolBox(padded_bbox, area, "digit"))
        elif is_dot:
            symbols.append(_SymbolBox(padded_bbox, area, "dot"))

    return symbols


def _group_digit_boxes_by_line(digit_boxes):
    if not digit_boxes:
        return []

    median_height = float(np.median([box.height for box in digit_boxes]))
    line_tolerance = max(10, median_height * 0.65)
    lines = []

    for box in sorted(digit_boxes, key=lambda item: item.center_y):
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


def _dot_belongs_to_line(dot_box, line):
    min_x = min(box.x for box in line)
    max_x = max(box.x + box.width for box in line)
    min_y = min(box.y for box in line)
    max_y = max(box.y + box.height for box in line)
    median_height = float(np.median([box.height for box in line]))

    within_y = min_y - median_height * 0.25 <= dot_box.center_y <= max_y + median_height * 0.25
    within_x = min_x - median_height * 0.5 <= dot_box.x <= max_x + median_height * 0.5
    return within_x and within_y


def _decimal_dots_for_line(dot_boxes, line):
    line = sorted(line, key=lambda item: item.x)
    median_height = float(np.median([box.height for box in line]))
    line_top = min(box.y for box in line)
    line_bottom = max(box.y + box.height for box in line)
    lower_half_start = line_top + (line_bottom - line_top) * 0.45

    decimal_dots = []
    for previous_digit, next_digit in zip(line, line[1:]):
        digit_gap = next_digit.x - (previous_digit.x + previous_digit.width)
        center_gap = (next_digit.x + next_digit.width / 2) - (
            previous_digit.x + previous_digit.width / 2
        )
        if digit_gap > median_height * 0.45 or center_gap > median_height * 1.2:
            continue

        previous_center_x = previous_digit.x + previous_digit.width / 2
        next_center_x = next_digit.x + next_digit.width / 2
        candidates = [
            dot
            for dot in dot_boxes
            if _dot_belongs_to_line(dot, line)
            and previous_center_x < dot.x + dot.width / 2 < next_center_x
            and dot.center_y >= lower_half_start
        ]
        if candidates:
            decimal_dots.append(max(candidates, key=lambda dot: dot.area))

    return decimal_dots


def _sort_symbols_reading_order(symbols):
    digit_boxes = [symbol for symbol in symbols if symbol.kind == "digit"]
    dot_boxes = [symbol for symbol in symbols if symbol.kind == "dot"]
    lines = _group_digit_boxes_by_line(digit_boxes)

    ordered_symbols = []
    for line in lines:
        line_symbols = list(line)
        line_symbols.extend(_decimal_dots_for_line(dot_boxes, line))
        ordered_symbols.extend(sorted(line_symbols, key=lambda item: item.x))

    return ordered_symbols


def _binarize_for_segmentation(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    threshold_type = cv2.THRESH_BINARY_INV if _background_is_light(gray_image) else cv2.THRESH_BINARY
    binary_image = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        51,
        11,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


def _find_symbol_boxes(binary_image):
    image_height, image_width = binary_image.shape
    raw_boxes = _find_raw_boxes(binary_image)
    symbols = _classify_boxes(raw_boxes, image_width, image_height)
    return _sort_symbols_reading_order(symbols)


def segment_symbols(image_path):
    image_path = Path(image_path)
    gray_image = _load_grayscale(image_path)
    binary_image = _binarize_for_segmentation(gray_image)

    segments = []
    for index, symbol in enumerate(_find_symbol_boxes(binary_image)):
        x, y, width, height = symbol.bbox
        crop = binary_image[y : y + height, x : x + width]
        centered = _resize_and_center(crop)
        segments.append(
            DigitSegment(
                index=index,
                bbox=symbol.bbox,
                image_array=centered,
                kind=symbol.kind,
            )
        )

    return segments


def segment_digit_arrays(image_path):
    return [segment for segment in segment_symbols(image_path) if segment.kind == "digit"]


def segment_digit_tensors(image_path):
    return [digit_array_to_tensor(segment.image_array) for segment in segment_digit_arrays(image_path)]


def save_segments(image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for segment in segment_symbols(image_path):
        output_path = output_dir / f"segment_{segment.index + 1:03}_{segment.kind}.png"
        Image.fromarray(segment.image_array).save(output_path)
        saved_paths.append(output_path)

    return saved_paths
