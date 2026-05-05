from pathlib import Path
from PIL import Image
import shutil
import uuid


# -----------------------------
# BASE DIRECTORY (project root relative)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR.parent / "samples" / "ux_raw"


# -----------------------------
# CORE: convert to PNG
# -----------------------------
def convert_to_png(file_path, output_dir=TARGET_DIR, rename_unique=True):
    """
    Convert image to PNG and save into dataset folder.
    Uses pathlib for all path handling.
    """

    file_path = Path(file_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # filename handling
    if rename_unique:
        filename = f"{uuid.uuid4()}.png"
    else:
        filename = file_path.stem + ".png"

    output_path = output_dir / filename

    try:
        img = Image.open(file_path).convert("RGB")
        img.save(output_path, "PNG")
    except Exception as e:
        raise RuntimeError(f"Image conversion failed: {e}")

    return output_path


# -----------------------------
# OPTIONAL: copy file as-is
# -----------------------------
def copy_file_to_dataset(file_path, output_dir=TARGET_DIR):
    """
    Copy file into dataset folder without conversion.
    """

    file_path = Path(file_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / file_path.name
    shutil.copy(file_path, destination)

    return destination


# -----------------------------
# OPTIONAL: validate image
# -----------------------------
def is_valid_image(file_path):
    """
    Check if file is a valid image.
    """

    file_path = Path(file_path)

    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False