import argparse
from pathlib import Path

import torch

from alphabet_row_segmentation import (
    PUNCTUATION_GLYPH_NAMES,
    format_alphabet_row_prediction,
    populate_font_library_from_alphabet_row,
    save_alphabet_row_segments,
    segment_alphabet_rows,
)
from corrections import save_correction
from english_postprocess import postprocess_text_lines, restore_sentence_case
from letter_corrections import save_letter_text_corrections
from model import DigitCNN, LetterCNN
from preprocessing import digit_array_to_tensor, preprocess_digit_image
from recognition_config import (
    DIGIT_LABEL_SET,
    LETTER_LABEL_SET,
    labels_for,
    model_path_for,
)
from render_handwriting import render_handwriting
from segmentation import save_segments, segment_symbols
from text_segmentation import save_text_segments, segment_text_lines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOW_CONFIDENCE_THRESHOLD = 0.50


def preprocess_image(image_path):
    return preprocess_digit_image(image_path)


def load_model(label_set=DIGIT_LABEL_SET):
    if label_set == DIGIT_LABEL_SET:
        model = DigitCNN()
    elif label_set == LETTER_LABEL_SET:
        model = LetterCNN()
    else:
        raise ValueError(f"Unsupported label set: {label_set}")

    model_path = model_path_for(label_set)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    except RuntimeError as exc:
        raise RuntimeError(
            "Model weights do not match the current architecture. "
            f"Please retrain the {label_set} model."
        ) from exc
    except FileNotFoundError as exc:
        if label_set == DIGIT_LABEL_SET:
            command = "python src/train_mnist.py"
        else:
            command = "python src/train_emnist_letters.py"
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please train it with: {command}"
        ) from exc
    model.eval()
    return model


def predict_tensor(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_index].item()
        return predicted_index, confidence


def label_from_prediction(predicted_index, label_set):
    return labels_for(label_set)[predicted_index]


def predict_single_character(model, image_path, label_set):
    image_tensor = preprocess_image(image_path)
    predicted_index, confidence = predict_tensor(model, image_tensor)
    return label_from_prediction(predicted_index, label_set), confidence


def predict_multiple_digits(model, image_path):
    segment_predictions = []

    for segment in segment_symbols(image_path):
        if segment.kind == "dot":
            segment_predictions.append(
                {
                    "index": segment.index + 1,
                    "kind": "dot",
                    "text": ".",
                    "confidence": None,
                    "bbox": segment.bbox,
                }
            )
        else:
            tensor = digit_array_to_tensor(segment.image_array)
            predicted_digit, confidence = predict_tensor(model, tensor)
            segment_predictions.append(
                {
                    "index": segment.index + 1,
                    "kind": "digit",
                    "text": str(predicted_digit),
                    "confidence": confidence,
                    "bbox": segment.bbox,
                }
            )

    return format_multi_prediction(segment_predictions), segment_predictions


def predict_multiple_letters(model, image_path):
    predicted_lines = []
    flat_predictions = []

    for line_segments in segment_text_lines(image_path):
        line_predictions = []
        for segment in line_segments:
            if segment.kind == "punctuation":
                prediction = {
                    "index": segment.index + 1,
                    "kind": "punctuation",
                    "text": segment.literal or ".",
                    "confidence": None,
                    "bbox": segment.bbox,
                    "separator_before": segment.separator_before,
                }
            else:
                tensor = digit_array_to_tensor(segment.image_array)
                predicted_index, confidence = predict_tensor(model, tensor)
                prediction = {
                    "index": segment.index + 1,
                    "kind": "character",
                    "text": label_from_prediction(predicted_index, LETTER_LABEL_SET),
                    "confidence": confidence,
                    "bbox": segment.bbox,
                    "separator_before": segment.separator_before,
                }
            line_predictions.append(prediction)
            flat_predictions.append(prediction)
        predicted_lines.append(line_predictions)

    return format_text_prediction(predicted_lines), flat_predictions


def predict_alphabet_rows(letter_model, digit_model, image_path, expected_text=None):
    predicted_rows = []
    flat_predictions = []

    for row_segments in segment_alphabet_rows(image_path, expected_text=expected_text):
        row_predictions = []
        for segment in row_segments:
            if segment.character in PUNCTUATION_GLYPH_NAMES:
                predicted_text = segment.character
                confidence = 1.0
            else:
                model = digit_model if (segment.character and segment.character.isdigit()) else letter_model
                label_set = DIGIT_LABEL_SET if (segment.character and segment.character.isdigit()) else LETTER_LABEL_SET
                tensor = digit_array_to_tensor(segment.image_array)
                predicted_index, confidence = predict_tensor(model, tensor)
                predicted_text = label_from_prediction(predicted_index, label_set)
                if segment.character and segment.character.islower() and predicted_text.isalpha():
                    predicted_text = predicted_text.lower()
            prediction = {
                "index": segment.index + 1,
                "kind": "character",
                "text": predicted_text,
                "confidence": confidence,
                "bbox": segment.bbox,
                "expected": segment.character,
                "line_index": segment.line_index,
            }
            row_predictions.append(prediction)
            flat_predictions.append(prediction)
        predicted_rows.append(row_predictions)

    return format_alphabet_row_prediction(predicted_rows), flat_predictions


def ask_for_correction(prediction):
    corrected = input(f"Correct label/text [{prediction}]: ").strip()
    return corrected or prediction


def format_confidence_label(confidence):
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return " [LOW CONFIDENCE]"
    return ""


def _segment_center_y(segment_prediction):
    x, y, width, height = segment_prediction["bbox"]
    return y + height / 2


def _segment_height(segment_prediction):
    return segment_prediction["bbox"][3]


def group_segment_predictions_by_line(segment_predictions):
    if not segment_predictions:
        return []

    digit_heights = [
        _segment_height(segment_prediction)
        for segment_prediction in segment_predictions
        if segment_prediction["kind"] == "digit"
    ]
    median_digit_height = (
        sorted(digit_heights)[len(digit_heights) // 2] if digit_heights else 40
    )
    line_tolerance = max(12, median_digit_height * 0.55)

    lines = []
    current_line = [segment_predictions[0]]
    current_center_y = _segment_center_y(segment_predictions[0])

    for segment_prediction in segment_predictions[1:]:
        center_y = _segment_center_y(segment_prediction)
        if abs(center_y - current_center_y) <= line_tolerance:
            current_line.append(segment_prediction)
            current_center_y = sum(_segment_center_y(item) for item in current_line) / len(current_line)
        else:
            lines.append(current_line)
            current_line = [segment_prediction]
            current_center_y = center_y

    lines.append(current_line)
    return lines


def format_line_tokens(line_predictions):
    tokens = []
    current_token = ""

    for segment_prediction in line_predictions:
        text = segment_prediction["text"]
        if text == ".":
            if current_token and not current_token.endswith("."):
                current_token += "."
            continue

        if not current_token:
            current_token = text
        elif current_token.endswith("."):
            current_token += text
            tokens.append(current_token)
            current_token = ""
        else:
            tokens.append(current_token)
            current_token = text

    if current_token:
        tokens.append(current_token)

    return tokens


def format_multi_prediction(segment_predictions):
    lines = group_segment_predictions_by_line(segment_predictions)
    line_text = []

    for line_predictions in lines:
        tokens = format_line_tokens(line_predictions)
        if tokens:
            line_text.append(" ".join(tokens))

    return "\n".join(line_text)


def format_text_prediction(predicted_lines):
    formatted_lines = []

    for line_predictions in predicted_lines:
        line_text = ""
        for prediction in line_predictions:
            if prediction.get("separator_before") and line_text:
                line_text += " "
            line_text += prediction["text"]
        if line_text:
            formatted_lines.append(line_text)

    return "\n".join(formatted_lines)


def postprocess_letter_prediction(predicted_text):
    lines = predicted_text.splitlines()
    corrected_text = "\n".join(postprocess_text_lines(lines))
    return restore_sentence_case(corrected_text)


def render_prediction_text(text, output_path=None, seed=1):
    return render_handwriting(text, output_path=output_path, seed=seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict handwritten character images.")
    parser.add_argument("image_path", help="Path to a handwritten character image.")
    parser.add_argument(
        "--label-set",
        choices=[DIGIT_LABEL_SET, LETTER_LABEL_SET],
        default=DIGIT_LABEL_SET,
        help="Choose whether to use the digit model or the letter model.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Segment and predict multiple characters from one image.",
    )
    parser.add_argument(
        "--text-mode",
        choices=["sentence", "alphabet-row"],
        default="sentence",
        help="For multi-letter images, choose sentence OCR or a simpler known-content alphabet-row mode.",
    )
    parser.add_argument(
        "--expected-text",
        help="Known row content for alphabet-row mode. Use newlines between rows if needed.",
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
    parser.add_argument(
        "--render-recognized",
        action="store_true",
        help="Render the predicted text back into a handwriting-style image using the font library.",
    )
    parser.add_argument(
        "--render-output",
        metavar="OUTPUT_PATH",
        help="Optional output path for the rendered handwriting image.",
    )
    parser.add_argument(
        "--render-seed",
        type=int,
        default=1,
        help="Optional random seed for deterministic handwriting rendering.",
    )
    parser.add_argument(
        "--populate-font-library",
        action="store_true",
        help="For alphabet-row images, export segmented glyphs into samples/font_letters and samples/font_digits as primary font samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    if args.multi:
        if args.text_mode == "alphabet-row":
            letter_model = load_model(LETTER_LABEL_SET)
            digit_model = load_model(DIGIT_LABEL_SET)
            prediction, segment_predictions = predict_alphabet_rows(
                letter_model,
                digit_model,
                image_path,
                expected_text=args.expected_text,
            )
        elif args.label_set == DIGIT_LABEL_SET:
            model = load_model(args.label_set)
            prediction, segment_predictions = predict_multiple_digits(model, image_path)
        else:
            model = load_model(args.label_set)
            prediction, segment_predictions = predict_multiple_letters(model, image_path)
            prediction = postprocess_letter_prediction(prediction)
        if not prediction:
            print("No character segments found.")
            return
        if args.text_mode == "alphabet-row":
            heading = "Predicted rows:"
        else:
            heading = "Predicted digits:" if args.label_set == DIGIT_LABEL_SET else "Predicted text:"
        print(heading)
        print(prediction)
        for segment_prediction in segment_predictions:
            if segment_prediction["kind"] in {"dot", "punctuation"}:
                print(f"Segment {segment_prediction['index']}: .")
            else:
                confidence_pct = segment_prediction["confidence"] * 100
                confidence_label = format_confidence_label(segment_prediction["confidence"])
                expected_suffix = ""
                if args.text_mode == "alphabet-row" and segment_prediction.get("expected"):
                    expected_suffix = f" | expected {segment_prediction['expected']}"
                print(
                    f"Segment {segment_prediction['index']}: "
                    f"{segment_prediction['text']} ({confidence_pct:.2f}%){confidence_label}{expected_suffix}"
                )
        mode = "multi"
    else:
        model = load_model(args.label_set)
        prediction, confidence = predict_single_character(model, image_path, args.label_set)
        confidence_label = format_confidence_label(confidence)
        item_name = "digit" if args.label_set == DIGIT_LABEL_SET else "letter"
        print(f"Predicted {item_name}: {prediction} ({confidence * 100:.2f}%){confidence_label}")
        mode = "single"

    if args.save_segments:
        if args.multi and args.text_mode == "alphabet-row":
            saved_paths = save_alphabet_row_segments(
                image_path,
                args.save_segments,
                expected_text=args.expected_text,
            )
        elif args.label_set == DIGIT_LABEL_SET:
            saved_paths = save_segments(image_path, args.save_segments)
        else:
            saved_paths = save_text_segments(image_path, args.save_segments)
        print(f"Saved {len(saved_paths)} segment image(s) to {args.save_segments}")

    if args.correct:
        corrected_label = ask_for_correction(prediction)
        labels_path = save_correction(image_path, prediction, corrected_label, mode=mode)
        print(f"Correction saved to {labels_path}")
        if args.label_set == LETTER_LABEL_SET and args.multi:
            letter_labels_path, saved_count, segment_count, corrected_count = save_letter_text_corrections(
                image_path,
                prediction,
                corrected_label,
            )
            print(
                f"Saved {saved_count} aligned letter segment correction(s) to {letter_labels_path} "
                f"from {segment_count} segment(s) and {corrected_count} corrected character(s)"
            )
        prediction = corrected_label

    if args.populate_font_library:
        saved_paths = populate_font_library_from_alphabet_row(
            image_path,
            expected_text=prediction if args.correct else args.expected_text,
        )
        print(f"Saved {len(saved_paths)} primary font sample(s) into the font library")

    if args.render_recognized:
        rendered_output_path = render_prediction_text(
            prediction,
            output_path=args.render_output,
            seed=args.render_seed,
        )
        print(f"Rendered predicted text to {rendered_output_path}")


if __name__ == "__main__":
    main()
