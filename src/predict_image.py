import argparse
from pathlib import Path

import torch
import torch.nn as nn

from corrections import save_correction
from preprocessing import digit_array_to_tensor, preprocess_digit_image
from segmentation import save_segments, segment_symbols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.pth"


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
    return preprocess_digit_image(image_path)


def load_model():
    model = SimpleNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict_tensor(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        return torch.argmax(output, dim=1).item()


def predict_single_digit(model, image_path):
    image_tensor = preprocess_image(image_path)
    return str(predict_tensor(model, image_tensor))


def predict_multiple_digits(model, image_path):
    predicted_text = []
    for segment in segment_symbols(image_path):
        if segment.kind == "dot":
            predicted_text.append(".")
        else:
            tensor = digit_array_to_tensor(segment.image_array)
            predicted_text.append(str(predict_tensor(model, tensor)))
    return "".join(predicted_text)


def ask_for_correction(prediction):
    corrected = input(f"Correct label/text [{prediction}]: ").strip()
    return corrected or prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Predict handwritten digit images.")
    parser.add_argument("image_path", help="Path to a handwritten digit image.")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Segment and predict multiple digits from one image.",
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        help="Prompt for the true label and save it to corrected_samples/labels.csv.",
    )
    parser.add_argument(
        "--save-segments",
        metavar="OUTPUT_DIR",
        help="Save preprocessed digit segments for debugging multi-digit images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    model = load_model()

    if args.multi:
        prediction = predict_multiple_digits(model, image_path)
        if not prediction:
            print("No digit segments found.")
            return
        print(f"Predicted digits: {prediction}")
        mode = "multi"
    else:
        prediction = predict_single_digit(model, image_path)
        print(f"Predicted digit: {prediction}")
        mode = "single"

    if args.save_segments:
        saved_paths = save_segments(image_path, args.save_segments)
        print(f"Saved {len(saved_paths)} segment image(s) to {args.save_segments}")

    if args.correct:
        corrected_label = ask_for_correction(prediction)
        labels_path = save_correction(image_path, prediction, corrected_label, mode=mode)
        print(f"Correction saved to {labels_path}")


if __name__ == "__main__":
    main()
