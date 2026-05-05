from pathlib import Path
from datetime import datetime
import csv


# -----------------------------
# BASE PATH (go UP from ux/ → project/)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"


# -----------------------------
# CATEGORIES
# -----------------------------
CATEGORIES = {
    "single_char": SAMPLES_DIR / "single_char",
    "random_handwriting": SAMPLES_DIR / "random_handwriting",
    "alphabet": SAMPLES_DIR / "alphabet",
    "ux_raw": SAMPLES_DIR / "ux_raw",
}

LABEL_FILE = SAMPLES_DIR / "labels.csv"


# -----------------------------
# INIT STRUCTURE
# -----------------------------
def init_dataset():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for path in CATEGORIES.values():
        path.mkdir(parents=True, exist_ok=True)

    if not LABEL_FILE.exists():
        with open(LABEL_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "category", "label", "timestamp"])


# -----------------------------
# SAVE FILE INTO CATEGORY
# -----------------------------
def save_to_dataset(file_path, category, label=None):
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    file_path = Path(file_path)
    target_dir = CATEGORIES[category]

    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{timestamp}_{file_path.stem}.png"

    output_path = target_dir / new_name

    output_path.write_bytes(file_path.read_bytes())

    # log
    with open(LABEL_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            str(output_path),
            category,
            label if label else "",
            timestamp
        ])

    return output_path