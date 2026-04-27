from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

DIGIT_LABEL_SET = "digits"
LETTER_LABEL_SET = "letters"

DIGIT_LABELS = [str(digit) for digit in range(10)]
LETTER_LABELS = [chr(ord("A") + index) for index in range(26)]

DIGIT_MODEL_PATH = MODELS_DIR / "mnist_model.pth"
LETTER_MODEL_PATH = MODELS_DIR / "emnist_letters_model.pth"


def labels_for(label_set):
    if label_set == DIGIT_LABEL_SET:
        return DIGIT_LABELS
    if label_set == LETTER_LABEL_SET:
        return LETTER_LABELS
    raise ValueError(f"Unsupported label set: {label_set}")


def model_path_for(label_set):
    if label_set == DIGIT_LABEL_SET:
        return DIGIT_MODEL_PATH
    if label_set == LETTER_LABEL_SET:
        return LETTER_MODEL_PATH
    raise ValueError(f"Unsupported label set: {label_set}")
