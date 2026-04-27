import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict_image import (
    format_multi_prediction,
    format_text_prediction,
    label_from_prediction,
    postprocess_letter_prediction,
    render_prediction_text,
)


def test_format_multi_prediction_groups_decimal_tokens_by_line():
    segment_predictions = [
        {"index": 1, "kind": "digit", "text": "1", "confidence": 0.9, "bbox": (10, 10, 10, 30)},
        {"index": 2, "kind": "digit", "text": "2", "confidence": 0.9, "bbox": (40, 10, 10, 30)},
        {"index": 3, "kind": "digit", "text": "3", "confidence": 0.9, "bbox": (10, 80, 10, 30)},
        {"index": 4, "kind": "dot", "text": ".", "confidence": None, "bbox": (28, 95, 6, 6)},
        {"index": 5, "kind": "digit", "text": "1", "confidence": 0.9, "bbox": (40, 80, 10, 30)},
        {"index": 6, "kind": "digit", "text": "4", "confidence": 0.9, "bbox": (80, 80, 10, 30)},
        {"index": 7, "kind": "dot", "text": ".", "confidence": None, "bbox": (98, 95, 6, 6)},
        {"index": 8, "kind": "digit", "text": "2", "confidence": 0.9, "bbox": (110, 80, 10, 30)},
    ]

    formatted = format_multi_prediction(segment_predictions)

    assert formatted == "1 2\n3.1 4.2"


def test_label_from_prediction_maps_letters_to_uppercase():
    assert label_from_prediction(0, "letters") == "A"
    assert label_from_prediction(25, "letters") == "Z"


def test_format_text_prediction_inserts_spaces_and_keeps_punctuation():
    predicted_lines = [
        [
            {"text": "A", "separator_before": False},
            {"text": "L", "separator_before": False},
            {"text": "I", "separator_before": False},
            {"text": "C", "separator_before": False},
            {"text": "E", "separator_before": False},
            {"text": "W", "separator_before": True},
            {"text": "A", "separator_before": False},
            {"text": "S", "separator_before": False},
        ],
        [
            {"text": "B", "separator_before": False},
            {"text": "A", "separator_before": False},
            {"text": "N", "separator_before": False},
            {"text": "K", "separator_before": False},
            {"text": ".", "separator_before": False},
        ],
    ]

    formatted = format_text_prediction(predicted_lines)

    assert formatted == "ALICE WAS\nBANK."


def test_postprocess_letter_prediction_corrects_common_words(monkeypatch):
    import predict_image

    monkeypatch.setattr(
        predict_image,
        "postprocess_text_lines",
        lambda lines: ["ALICE WAS BEGINNING TO GET VERY TIRED OF SITTING BY HER", "SISTER ON THE BANK."],
    )
    monkeypatch.setattr(
        predict_image,
        "restore_sentence_case",
        lambda text: "Alice was beginning to get very tired of sitting by her\nsister on the bank.",
    )

    corrected = postprocess_letter_prediction("ALICE WAS BEGINNING W GET VORY TINED OI SITTING NY HER\nSISTER AN THE BAWK.")

    assert corrected == "Alice was beginning to get very tired of sitting by her\nsister on the bank."


def test_render_prediction_text_delegates_to_render_handwriting(monkeypatch):
    import predict_image

    captured = {}

    def fake_render_handwriting(text, output_path=None, seed=None):
        captured["text"] = text
        captured["output_path"] = output_path
        captured["seed"] = seed
        return "output/fake.png"

    monkeypatch.setattr(predict_image, "render_handwriting", fake_render_handwriting)

    rendered_path = render_prediction_text("HELLO?", output_path="output/hello.png", seed=7)

    assert rendered_path == "output/fake.png"
    assert captured == {
        "text": "HELLO?",
        "output_path": "output/hello.png",
        "seed": 7,
    }
