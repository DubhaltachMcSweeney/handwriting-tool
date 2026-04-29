# Sample Naming Guide

Use this folder for project input images. Keep raw user images separate from processed images so tests and model debugging stay clear.

## Folder Structure

- `raw/digits/`: single handwritten digit images before preprocessing.
- `raw/multi_digit/`: images containing multiple digits, such as `123` or `9085`.
- `raw/letters/`: handwritten letter images for future EMNIST/OCR experiments.
- `processed/digits/`: optional debug outputs generated after preprocessing.

## File Naming

Use lowercase names with underscores:

```text
digit_<label>_<number>_raw.png
digit_unknown_<number>_raw.png
multi_digit_<text>_<number>_raw.png
letter_<label>_<number>_raw.png
```

Examples:

```text
raw/digits/digit_3_001_raw.png
raw/digits/digit_8_001_raw.png
raw/digits/digit_unknown_001_raw.png
raw/multi_digit/multi_digit_123_001_raw.png
raw/letters/letter_a_001_raw.png
```

Use `unknown` when you have not confirmed the true label yet. Do not name files using only the model prediction unless you have checked the image yourself.
