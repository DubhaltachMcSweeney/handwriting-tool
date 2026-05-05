from pathlib import Path
import shutil
import re

# -----------------------------
# BASE PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples" / "raw"

DIGITS_DIR = SAMPLES_DIR / "digits"
LETTERS_DIR = SAMPLES_DIR / "letters"
MULTI_DIGIT_DIR = SAMPLES_DIR / "multi_digit"


# -----------------------------
# HELPERS
# -----------------------------
def _ensure_dirs():
    DIGITS_DIR.mkdir(parents=True, exist_ok=True)
    LETTERS_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_DIGIT_DIR.mkdir(parents=True, exist_ok=True)


def _get_next_index(folder, pattern):
    """
    Finds next available index based on existing filenames.
    pattern = regex with capturing group for index
    """
    max_index = 0

    for file in folder.iterdir():
        match = re.match(pattern, file.name)
        if match:
            try:
                idx = int(match.group(1))
                max_index = max(max_index, idx)
            except ValueError:
                continue

    return max_index + 1


# -----------------------------
# MAIN SAVE FUNCTION
# -----------------------------
def save_to_pipeline(file_path, task_type, label=None):
    """
    Saves file into samples/raw/... following strict naming rules.

    task_type:
        digit
        letter
        sentence
        multi_digit
    """

    _ensure_dirs()

    file_path = Path(file_path)

    # -------------------------
    # DIGIT
    # -------------------------
    if task_type == "digit":
        folder = DIGITS_DIR

        if label:
            pattern = r"digit_[^_]+_(\d+)_raw\.png"
            idx = _get_next_index(folder, pattern)
            filename = f"digit_{label}_{idx:03d}_raw.png"
        else:
            pattern = r"digit_unknown_(\d+)_raw\.png"
            idx = _get_next_index(folder, pattern)
            filename = f"digit_unknown_{idx:03d}_raw.png"

    # -------------------------
    # LETTER
    # -------------------------
    elif task_type == "letter":
        folder = LETTERS_DIR

        if not label:
            raise ValueError("Letter samples require a label (e.g. 'a')")

        pattern = r"letter_[^_]+_(\d+)_raw\.png"
        idx = _get_next_index(folder, pattern)

        filename = f"letter_{label}_{idx:03d}_raw.png"

    # -------------------------
    # SENTENCE
    # -------------------------
    elif task_type == "sentence":
        folder = LETTERS_DIR

        if not label:
            raise ValueError("Sentence samples require a description label")

        # normalize label (safe filename)
        safe_label = label.lower().replace(" ", "_")

        pattern = r"sentence_[^_]+_(\d+)_raw\.(png|jpeg|jpg)"
        idx = _get_next_index(folder, pattern)

        filename = f"sentence_{safe_label}_{idx:03d}_raw.jpeg"

    # -------------------------
    # MULTI DIGIT
    # -------------------------
    elif task_type == "multi_digit":
        folder = MULTI_DIGIT_DIR

        if not label:
            raise ValueError("Multi-digit samples require label text (e.g. '123')")

        pattern = r"multi_digit_[^_]+_(\d+)_raw\.png"
        idx = _get_next_index(folder, pattern)

        filename = f"multi_digit_{label}_{idx:03d}_raw.png"

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # -------------------------
    # SAVE FILE
    # -------------------------
    output_path = folder / filename
    shutil.copy(file_path, output_path)

    return output_path