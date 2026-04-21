import sys
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0)
    return tensor


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        return

    image_path = sys.argv[1]

    model = SimpleNN()
    model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
    model.eval()

    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()

    print(f"Predicted digit: {predicted}")


if __name__ == "__main__":
    main()