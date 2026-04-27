from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


IMAGE_SIZE = 28
PADDING = 4


def _load_grayscale(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image.convert("RGBA"))
        return np.array(image.convert("L"))
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or invalid image file: {image_path}") from exc


def _background_is_light(gray_image):
    border_pixels = np.concatenate(
        [
            gray_image[0, :],
            gray_image[-1, :],
            gray_image[:, 0],
            gray_image[:, -1],
        ]
    )
    return border_pixels.mean() > 127


def _make_foreground_light(gray_image):
    if _background_is_light(gray_image):
        return 255 - gray_image
    return gray_image


def _binarize(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def _crop_to_content(binary_image):
    points = cv2.findNonZero(binary_image)
    if points is None:
        return binary_image

    x, y, width, height = cv2.boundingRect(points)
    return binary_image[y : y + height, x : x + width]


def _normalize_digit_strokes(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    normalized = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    foreground_ratio = float(np.count_nonzero(normalized)) / normalized.size
    if foreground_ratio < 0.18:
        normalized = cv2.dilate(normalized, kernel, iterations=1)

    return _crop_to_content(normalized)


def _resize_and_center(binary_image, image_size=IMAGE_SIZE, padding=PADDING):
    height, width = binary_image.shape
    inner_size = image_size - 2 * padding

    scale = min(inner_size / width, inner_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(binary_image, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    top = (image_size - new_height) // 2
    left = (image_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized

    return canvas


def digit_array_to_tensor(image_array):
    image = Image.fromarray(image_array.astype(np.uint8))
    return transforms.ToTensor()(image).unsqueeze(0)


def preprocess_digit_array(image_path):
    image_path = Path(image_path)
    gray_image = _load_grayscale(image_path)
    return preprocess_digit_array_from_gray(gray_image)


def preprocess_digit_array_from_gray(gray_image):
    foreground_light = _make_foreground_light(gray_image)
    binary_image = _binarize(foreground_light)
    cropped_image = _normalize_digit_strokes(_crop_to_content(binary_image))
    return _resize_and_center(cropped_image)


def preprocess_digit_image(image_path):
    centered_image = preprocess_digit_array(image_path)
    return digit_array_to_tensor(centered_image)


def save_processed_digit(image_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centered_image = preprocess_digit_array(image_path)
    Image.fromarray(centered_image).save(output_path)
    return output_path
