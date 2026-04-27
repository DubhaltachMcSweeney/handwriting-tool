from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from model import LetterCNN
from recognition_config import LETTER_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _fix_emnist_orientation(image):
    image = TF.rotate(image, -90)
    image = TF.hflip(image)
    return image


def main():
    base_transforms = [
        transforms.Lambda(_fix_emnist_orientation),
    ]
    train_transform = transforms.Compose(
        base_transforms
        + [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
            ),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(base_transforms + [transforms.ToTensor()])

    target_transform = lambda label: label - 1

    train_dataset = datasets.EMNIST(
        root=DATA_DIR,
        split="letters",
        train=True,
        download=True,
        transform=train_transform,
        target_transform=target_transform,
    )

    test_dataset = datasets.EMNIST(
        root=DATA_DIR,
        split="letters",
        train=False,
        download=True,
        transform=test_transform,
        target_transform=target_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = LetterCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    LETTER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LETTER_MODEL_PATH)
    print(f"Model saved to {LETTER_MODEL_PATH}")


if __name__ == "__main__":
    main()
