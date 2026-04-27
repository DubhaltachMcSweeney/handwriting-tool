import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from model import DigitCNN
from preprocessing import preprocess_digit_array
from segmentation import segment_symbols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.pth"
LABELS_PATH = PROJECT_ROOT / "corrected_samples" / "labels.csv"
MULTI_DIGIT_DIR = PROJECT_ROOT / "samples" / "raw" / "multi_digit"

REHEARSAL_SAMPLES = 2000
CORRECTION_REPEATS = 128
MULTI_DIGIT_REPEATS = 32


def _digit_label(text):
    text = str(text).strip()
    if len(text) != 1 or not text.isdigit():
        return None
    return int(text)


def load_single_digit_corrections(labels_path=LABELS_PATH):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return []

    samples = []
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("mode") != "single":
                continue

            label = _digit_label(row.get("corrected_label", ""))
            image_path = row.get("saved_image_path")
            if label is None or not image_path:
                continue

            image_path = Path(image_path)
            if not image_path.exists():
                continue

            samples.append({"image_path": image_path, "label": label})

    return samples


def load_labeled_multi_digit_samples(multi_digit_dir=MULTI_DIGIT_DIR):
    multi_digit_dir = Path(multi_digit_dir)
    if not multi_digit_dir.exists():
        return []

    samples = []
    pattern = re.compile(r"^multi_digit_(.+)_(\d+)_raw$")

    for image_path in sorted(multi_digit_dir.glob("multi_digit_*_raw.*")):
        match = pattern.match(image_path.stem)
        if not match:
            continue

        label_text = match.group(1)
        if "unknown" in label_text:
            continue

        expected_digits = [int(char) for char in label_text if char.isdigit()]
        if not expected_digits:
            continue

        digit_segments = [
            segment for segment in segment_symbols(image_path) if segment.kind == "digit"
        ]
        if len(digit_segments) != len(expected_digits):
            print(
                f"Skipping {image_path.name}: expected {len(expected_digits)} digits, "
                f"got {len(digit_segments)} segments"
            )
            continue

        for segment, label in zip(digit_segments, expected_digits):
            samples.append({"image_array": segment.image_array, "label": label})

    return samples


class CorrectedDigitDataset(Dataset):
    def __init__(self, samples, repeats=1, transform=None):
        self.samples = list(samples)
        self.repeats = max(1, int(repeats))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        if "image_array" in sample:
            image_array = sample["image_array"]
        else:
            image_array = preprocess_digit_array(sample["image_path"])
        image = Image.fromarray(image_array.astype(np.uint8))
        return self.transform(image), sample["label"]


def build_rehearsal_dataset():
    rehearsal_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
        ]
    )
    mnist_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=rehearsal_transform,
    )

    sample_count = min(REHEARSAL_SAMPLES, len(mnist_dataset))
    indices = torch.randperm(len(mnist_dataset))[:sample_count].tolist()
    return Subset(mnist_dataset, indices)


def fine_tune():
    correction_samples = load_single_digit_corrections()
    multi_digit_samples = load_labeled_multi_digit_samples()
    if not correction_samples and not multi_digit_samples:
        print(
            "No usable correction samples found in corrected_samples/labels.csv "
            "or labeled multi-digit images."
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

    if correction_samples:
        fine_tune_datasets.append(
            CorrectedDigitDataset(
                correction_samples,
                repeats=CORRECTION_REPEATS,
                transform=correction_transform,
            )
        )

    if multi_digit_samples:
        fine_tune_datasets.append(
            CorrectedDigitDataset(
                multi_digit_samples,
                repeats=MULTI_DIGIT_REPEATS,
                transform=correction_transform,
            )
        )

    train_dataset = ConcatDataset(fine_tune_datasets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
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

        print(
            f"Fine-tune epoch {epoch + 1}/{epochs}, "
            f"loss: {running_loss / len(train_loader):.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Fine-tuned model saved to {MODEL_PATH}")
    print(f"Used {len(correction_samples)} single-digit correction sample(s)")
    print(f"Used {len(multi_digit_samples)} labeled multi-digit segment sample(s)")


if __name__ == "__main__":
    fine_tune()
