# Demo Script

This file is a practical walkthrough for demonstrating the project live.

Recommended demo sample:

- `samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg`

Recommended supporting sample:

- `samples/raw/multi_digit/multi_digit_unknown_001_raw.png`

## Demo Goal

Show the full interactive loop:

```text
handwritten image -> recognition -> correction -> fine-tuning -> improved recognition -> personalised rendering
```

## Before You Start

Open the project folder and activate the virtual environment:

```bash
cd /Users/yuxinyu/handwriting-tool
source venv/bin/activate
```

## Step 1. Show the Input Image

Explain that the input is a handwritten sentence written by the user.

Demo file:

```text
samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg
```

## Step 2. Run Initial Recognition

Command:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
```

What to say:

- the image is segmented into candidate characters
- the letter model predicts each character
- the output is then repaired with English post-processing

## Step 3. Show User Correction

Command:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --correct
```

When prompted, paste the correct sentence:

```text
Alice was beginning to get very tired of sitting by her sister on the bank.
```

What to say:

- the user can correct the predicted sentence
- the system saves the full correction
- the system also aligns the corrected sentence to segmented character images

## Step 4. Fine-Tune the Letter Model

Command:

```bash
venv/bin/python src/fine_tune_letters.py
```

What to say:

- the system uses corrected user handwriting samples
- this adapts the letter model to the user’s own handwriting style

## Step 5. Recognise the Same Sentence Again

Command:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
```

What to say:

- this is the second recognition pass after adaptation
- the output should be closer to the user’s true sentence

## Step 6. Render the Recognised Text Back into Handwriting

Command:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --render-recognized --render-output output/demo_rendered_sentence.png
```

What to say:

- the recognised text is now turned back into a handwriting-style image
- this uses the stored personal character library

Output file:

```text
output/demo_rendered_sentence.png
```

## Optional Step 7. Show a Stronger Digit Result

Command:

```bash
venv/bin/python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png
```

What to say:

- digit recognition is currently the strongest part of the system
- this helps show that the project works well in a constrained setting

## Step 8. Generate TTF

Command:

```bash
python src/font_generator.py --output output/MyHandwriting.ttf --family-name "MyHandwriting"
```


## Recommended Final Message

Close the demo with:

```text
This project is not just a handwriting recogniser. It is an interactive pipeline that supports recognition, correction, user-specific adaptation, and personalised handwriting-style rendering.
```
