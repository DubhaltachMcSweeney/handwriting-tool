import argparse
from pathlib import Path

from alphabet_row_segmentation import (
    populate_font_library_from_alphabet_row,
    save_alphabet_row_segments,
)
from font_generation import OUTPUT_TTF, build_font
from predict_image import (
    DIGIT_LABEL_SET,
    LETTER_LABEL_SET,
    LOW_CONFIDENCE_THRESHOLD,
    format_confidence_label,
    load_model,
    predict_alphabet_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview a known alphabet-row image, export the segmented glyphs into the "
            "font library, and build a TTF from those samples."
        )
    )
    parser.add_argument(
        "image_path",
        help=(
            "Path to the alphabet-row image containing uppercase, lowercase, and digit rows, "
            "optionally followed by a punctuation row like . , ? ! or . , ? ! ' \" : ; - ( )."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_TTF,
        help="Output TTF path.",
    )
    parser.add_argument(
        "--family-name",
        default="MyHandwriting",
        help="Font family name to embed in the generated TTF.",
    )
    parser.add_argument(
        "--expected-text",
        help=(
            "Optional known row content. If omitted, the default template is "
            "A-Z / a-z / 1234567890, with an automatic fourth row of . , ? ! or "
            ". , ? ! ' \" : ; - ( ) when present."
        ),
    )
    parser.add_argument(
        "--save-segments",
        metavar="OUTPUT_DIR",
        help="Optional directory for saving debug segment crops before font export.",
    )
    return parser.parse_args()


def _print_preview(prediction, segment_predictions):
    print("Preview prediction:")
    print(prediction)
    for segment_prediction in segment_predictions:
        confidence_pct = segment_prediction["confidence"] * 100
        confidence_label = format_confidence_label(segment_prediction["confidence"])
        expected_suffix = ""
        if segment_prediction.get("expected"):
            expected_suffix = f" | expected {segment_prediction['expected']}"
        print(
            f"Segment {segment_prediction['index']}: "
            f"{segment_prediction['text']} ({confidence_pct:.2f}%){confidence_label}{expected_suffix}"
        )


def _summarize_preview(segment_predictions):
    total = len(segment_predictions)
    mismatches = [
        prediction
        for prediction in segment_predictions
        if prediction.get("expected") and prediction["text"] != prediction["expected"]
    ]
    low_confidence = [
        prediction
        for prediction in segment_predictions
        if prediction["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]
    print(
        f"Preview summary: {total} segment(s), "
        f"{len(mismatches)} mismatch(es), "
        f"{len(low_confidence)} low-confidence segment(s)"
    )


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    letter_model = load_model(LETTER_LABEL_SET)
    digit_model = load_model(DIGIT_LABEL_SET)

    prediction, segment_predictions = predict_alphabet_rows(
        letter_model,
        digit_model,
        image_path,
        expected_text=args.expected_text,
    )
    _print_preview(prediction, segment_predictions)
    _summarize_preview(segment_predictions)

    if args.save_segments:
        saved_segment_paths = save_alphabet_row_segments(
            image_path,
            args.save_segments,
            expected_text=args.expected_text,
        )
        print(f"Saved {len(saved_segment_paths)} debug segment image(s) to {args.save_segments}")

    exported = populate_font_library_from_alphabet_row(
        image_path,
        expected_text=args.expected_text,
    )
    print(f"Exported {len(exported)} primary font sample(s) into the font library")

    build_font(output_path=args.output, family_name=args.family_name)
    print(f"Saved {args.output}")
    print("Done: previewed the alphabet row, populated the font library, and built the TTF.")


if __name__ == "__main__":
    main()
