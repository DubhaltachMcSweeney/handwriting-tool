import csv
from datetime import datetime
from pathlib import Path

from PIL import Image

from text_segmentation import segment_text_lines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = PROJECT_ROOT / "corrected_samples" / "letter_segments"
IMAGES_DIR = CORRECTIONS_DIR / "images"
LABELS_PATH = CORRECTIONS_DIR / "labels.csv"

FIELDNAMES = [
    "timestamp",
    "original_path",
    "segment_index",
    "saved_image_path",
    "predicted_char",
    "corrected_char",
]


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_text_characters(text):
    normalized = []
    for char in str(text):
        if char.isalpha():
            normalized.append(char.upper())
        elif char == ".":
            normalized.append(".")
    return normalized


def align_characters(predicted_chars, corrected_chars):
    predicted_chars = list(predicted_chars)
    corrected_chars = list(corrected_chars)
    rows = len(predicted_chars) + 1
    cols = len(corrected_chars) + 1

    dp = [[0] * cols for _ in range(rows)]
    back = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if predicted_chars[i - 1] == corrected_chars[j - 1] else 1
            candidates = [
                (dp[i - 1][j - 1] + substitution_cost, "match"),
                (dp[i - 1][j] + 1, "delete"),
                (dp[i][j - 1] + 1, "insert"),
            ]
            dp[i][j], back[i][j] = min(candidates, key=lambda item: item[0])

    alignment = []
    i = len(predicted_chars)
    j = len(corrected_chars)
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "match":
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "delete":
            alignment.append((i - 1, None))
            i -= 1
        elif action == "insert":
            alignment.append((None, j - 1))
            j -= 1
        else:
            break

    return list(reversed(alignment))


def _flat_segments(image_path):
    return [segment for line in segment_text_lines(image_path) for segment in line]


def save_letter_text_corrections(image_path, predicted_text, corrected_text):
    image_path = Path(image_path)
    segments = _flat_segments(image_path)

    predicted_chars = []
    character_segments = []
    for segment in segments:
        if segment.kind == "character":
            predicted_chars.append("?")
            character_segments.append(segment)
        elif segment.literal == ".":
            predicted_chars.append(".")
            character_segments.append(segment)

    normalized_prediction = normalize_text_characters(predicted_text)
    normalized_correction = normalize_text_characters(corrected_text)

    if len(normalized_prediction) != len(character_segments):
        normalized_prediction = [
            segment.literal if segment.kind == "punctuation" else "?"
            for segment in character_segments
        ]

    alignment = align_characters(normalized_prediction, normalized_correction)

    timestamp = _timestamp()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    saved_rows = []
    for predicted_index, corrected_index in alignment:
        if predicted_index is None or corrected_index is None:
            continue

        segment = character_segments[predicted_index]
        corrected_char = normalized_correction[corrected_index]
        predicted_char = normalized_prediction[predicted_index]

        if segment.kind != "character":
            continue
        if not corrected_char.isalpha():
            continue

        saved_image_path = IMAGES_DIR / (
            f"{image_path.stem}_segment_{segment.index + 1:03}_{timestamp}.png"
        )
        Image.fromarray(segment.image_array).save(saved_image_path)

        saved_rows.append(
            {
                "timestamp": timestamp,
                "original_path": str(image_path),
                "segment_index": str(segment.index + 1),
                "saved_image_path": str(saved_image_path),
                "predicted_char": predicted_char,
                "corrected_char": corrected_char,
            }
        )

    file_exists = LABELS_PATH.exists()
    with LABELS_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in saved_rows:
            writer.writerow(row)

    return LABELS_PATH, len(saved_rows), len(character_segments), len(normalized_correction)
