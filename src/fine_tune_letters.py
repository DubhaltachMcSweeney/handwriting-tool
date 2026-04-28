import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from letter_corrections import LABELS_PATH as LETTER_SEGMENT_LABELS_PATH
from model import LetterCNN
from preprocessing import preprocess_digit_array
from recognition_config import LETTER_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GENERAL_LABELS_PATH = PROJECT_ROOT / "corrected_samples" / "labels.csv"
RAW_LETTER_DIR = PROJECT_ROOT / "samples" / "raw" / "letters"
FONT_LETTERS_DIR = PROJECT_ROOT / "samples" / "font_letters"

REHEARSAL_SAMPLES = 3000
CORRECTION_REPEATS = 96
SEGMENT_REPEATS = 48
RAW_LETTER_REPEATS = 32
FONT_LETTER_REPEATS = 24


def _letter_label(text):
    text = str(text).strip()
    if len(text) != 1 or not text.isalpha():
        return None
    return ord(text.upper()) - ord("A")


def load_single_letter_corrections(labels_path=GENERAL_LABELS_PATH):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return []

    samples = []
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("mode") != "single":
                continue

            label = _letter_label(row.get("corrected_label", ""))
            image_path = row.get("saved_image_path")
            if label is None or not image_path:
                continue

            image_path = Path(image_path)
            if not image_path.exists():
                continue

            samples.append({"image_path": image_path, "label": label})

    return samples


def load_segment_letter_corrections(labels_path=LETTER_SEGMENT_LABELS_PATH):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return []

    samples = []
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = _letter_label(row.get("corrected_char", ""))
            image_path = row.get("saved_image_path")
            if label is None or not image_path:
                continue

            image_path = Path(image_path)
            if not image_path.exists():
                continue

            samples.append({"image_path": image_path, "label": label})

    return samples


def load_raw_letter_samples(raw_letter_dir=RAW_LETTER_DIR):
    raw_letter_dir = Path(raw_letter_dir)
    if not raw_letter_dir.exists():
        return []

    samples = []
    pattern = re.compile(r"^letter_([A-Za-z])_(\d+)_raw$")

    for image_path in sorted(raw_letter_dir.glob("letter_*_raw.*")):
        match = pattern.match(image_path.stem)
        if not match:
            continue

        label = _letter_label(match.group(1))
        if label is None:
            continue
        samples.append({"image_path": image_path, "label": label})

    return samples


def load_font_letter_samples(font_letters_dir=FONT_LETTERS_DIR):
    font_letters_dir = Path(font_letters_dir)
    if not font_letters_dir.exists():
        return []

    samples = []
    for case_folder in ("uppercase", "lowercase"):
        case_dir = font_letters_dir / case_folder
        if not case_dir.exists():
            continue

        for char_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
            character = char_dir.name
            label = _letter_label(character)
            if label is None:
                continue

            for image_path in sorted(char_dir.glob("letter_*.png")):
                if image_path.is_file():
                    samples.append({"image_path": image_path, "label": label})

    return samples


class CorrectedLetterDataset(Dataset):
    def __init__(self, samples, repeats=1, transform=None):
        self.samples = list(samples)
        self.repeats = max(1, int(repeats))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        image_array = preprocess_digit_array(sample["image_path"])
        image = Image.fromarray(image_array.astype(np.uint8))
        return self.transform(image), sample["label"]


def _fix_emnist_orientation(image):
    image = TF.rotate(image, -90)
    image = TF.hflip(image)
    return image


def build_rehearsal_dataset():
    rehearsal_transform = transforms.Compose(
        [
            transforms.Lambda(_fix_emnist_orientation),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
        ]
    )

    target_transform = lambda label: label - 1
    emnist_dataset = datasets.EMNIST(
        root=DATA_DIR,
        split="letters",
        train=True,
        download=False,
        transform=rehearsal_transform,
        target_transform=target_transform,
    )

    sample_count = min(REHEARSAL_SAMPLES, len(emnist_dataset))
    indices = torch.randperm(len(emnist_dataset))[:sample_count].tolist()
    return Subset(emnist_dataset, indices)


def fine_tune():
    single_samples = load_single_letter_corrections()
    segment_samples = load_segment_letter_corrections()
    raw_letter_samples = load_raw_letter_samples()
    font_letter_samples = load_font_letter_samples()

    if not single_samples and not segment_samples and not raw_letter_samples and not font_letter_samples:
        print(
            "No usable letter correction samples found in corrected_samples/labels.csv, "
            "corrected_samples/letter_segments/labels.csv, samples/raw/letters, "
            "or samples/font_letters."
        )
        return

    correction_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
        ]
    )

    fine_tune_datasets = [build_rehearsal_dataset()]

    if single_samples:
        fine_tune_datasets.append(
            CorrectedLetterDataset(
                single_samples,
                repeats=CORRECTION_REPEATS,
                transform=correction_transform,
            )
        )

    if segment_samples:
        fine_tune_datasets.append(
            CorrectedLetterDataset(
                segment_samples,
                repeats=SEGMENT_REPEATS,
                transform=correction_transform,
            )
        )

    if raw_letter_samples:
        fine_tune_datasets.append(
            CorrectedLetterDataset(
                raw_letter_samples,
                repeats=RAW_LETTER_REPEATS,
                transform=correction_transform,
            )
        )

    if font_letter_samples:
        fine_tune_datasets.append(
            CorrectedLetterDataset(
                font_letter_samples,
                repeats=FONT_LETTER_REPEATS,
                transform=correction_transform,
            )
        )

    train_dataset = ConcatDataset(fine_tune_datasets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LetterCNN()
    model.load_state_dict(torch.load(LETTER_MODEL_PATH, map_location=torch.device("cpu")))
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 3

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Fine-tune epoch {epoch + 1}/{epochs}, loss: {running_loss / len(train_loader):.4f}")

    LETTER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LETTER_MODEL_PATH)
    print(f"Fine-tuned model saved to {LETTER_MODEL_PATH}")
    print(f"Used {len(single_samples)} single-letter correction sample(s)")
    print(f"Used {len(segment_samples)} aligned sentence segment sample(s)")
    print(f"Used {len(raw_letter_samples)} labeled raw letter sample(s)")
    print(f"Used {len(font_letter_samples)} font library letter sample(s)")


if __name__ == "__main__":
    fine_tune()
