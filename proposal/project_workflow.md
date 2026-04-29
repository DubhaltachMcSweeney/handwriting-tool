# Project Workflow and Presentation Guide

This file is the best single place to open if someone wants to understand:

- what the project currently does
- which files implement each part
- which commands to run for each workflow
- which extra presentation documents exist

## 1. What the support files are for

The following files are presentation and reporting support documents:

- [project_proposal.md](/Users/yuxinyu/handwriting-tool/proposal/project_proposal.md)  
  Original proposal and scope document.

- [evaluation_summary.md](/Users/yuxinyu/handwriting-tool/proposal/evaluation_summary.md)  
  A compact evaluation summary with representative results, before/after examples, and OCR baseline comparison.

- [demo_script.md](/Users/yuxinyu/handwriting-tool/proposal/demo_script.md)  
  A live-demo walkthrough explaining what to show and what to say.

- [demo_commands.md](/Users/yuxinyu/handwriting-tool/proposal/demo_commands.md)  
  A short command-only checklist for live use.

- [demo_presentation_notes.md](/Users/yuxinyu/handwriting-tool/proposal/demo_presentation_notes.md)  
  A short English speaking script for the presentation.

## 2. If the lecturer wants to understand the full project

Use this order:

1. Read [project_proposal.md](/Users/yuxinyu/handwriting-tool/proposal/project_proposal.md) for the original aim and scope.
2. Read [project_workflow.md](/Users/yuxinyu/handwriting-tool/proposal/project_workflow.md) for the current implementation and full command flow.
3. Read [evaluation_summary.md](/Users/yuxinyu/handwriting-tool/proposal/evaluation_summary.md) for the main results.

## 3. Main implementation files

### Recognition and inference

- [src/predict_image.py](/Users/yuxinyu/handwriting-tool/src/predict_image.py)  
  Main entry point for prediction, correction, and rendering recognised text.

- [src/model.py](/Users/yuxinyu/handwriting-tool/src/model.py)  
  CNN model definitions for digits and letters.

- [src/recognition_config.py](/Users/yuxinyu/handwriting-tool/src/recognition_config.py)  
  Label sets and model file paths.

### Preprocessing and segmentation

- [src/preprocessing.py](/Users/yuxinyu/handwriting-tool/src/preprocessing.py)  
  Image preprocessing for digits and segmented character crops.

- [src/segmentation.py](/Users/yuxinyu/handwriting-tool/src/segmentation.py)  
  Multi-digit segmentation with decimal handling.

- [src/text_segmentation.py](/Users/yuxinyu/handwriting-tool/src/text_segmentation.py)  
  Sentence/text segmentation for handwritten letters and sentence images.

### Training and fine-tuning

- [src/train_mnist.py](/Users/yuxinyu/handwriting-tool/src/train_mnist.py)  
  Digit model training.

- [src/train_emnist_letters.py](/Users/yuxinyu/handwriting-tool/src/train_emnist_letters.py)  
  Letter model training.

- [src/fine_tune_corrections.py](/Users/yuxinyu/handwriting-tool/src/fine_tune_corrections.py)  
  Digit fine-tuning with user corrections and labelled multi-digit samples.

- [src/fine_tune_letters.py](/Users/yuxinyu/handwriting-tool/src/fine_tune_letters.py)  
  Letter fine-tuning with aligned sentence corrections, raw letter samples, and `font_letters`.

### Correction and post-processing

- [src/corrections.py](/Users/yuxinyu/handwriting-tool/src/corrections.py)  
  Saves general correction records.

- [src/letter_corrections.py](/Users/yuxinyu/handwriting-tool/src/letter_corrections.py)  
  Aligns corrected sentence text to character segments and stores letter-level correction samples.

- [src/english_postprocess.py](/Users/yuxinyu/handwriting-tool/src/english_postprocess.py)  
  English word repair, phrase repair, and sentence-case restoration.

### Handwriting-style rendering

- [src/render_handwriting.py](/Users/yuxinyu/handwriting-tool/src/render_handwriting.py)  
  Renders typed or recognised text using the personal digit and letter libraries.

### Font generation

- [src/font_generator.py](/Users/yuxinyu/handwriting-tool/src/font_generation.py)  
  Vectorises the personal character library and assembles it into an installable TrueType (`.ttf`) font.

## 4. Core workflows

Each command block below now includes:

- **Purpose**: what the command does
- **Typical output**: what you should expect to see
- **Files produced**: which files or folders are updated

### A. Digit model training

**Purpose**

Train the digit recogniser on MNIST and save the trained model.

Train the digit recogniser:

```bash
cd /Users/yuxinyu/handwriting-tool
source venv/bin/activate
venv/bin/python src/train_mnist.py
```

Output model:

```text
models/mnist_model.pth
```

**Typical output**

```text
Epoch 1/...
Epoch 2/...
Test Accuracy: ...
Model saved to .../models/mnist_model.pth
```

### B. Letter model training

**Purpose**

Train the letter recogniser on EMNIST Letters and save the trained model.

Train the letter recogniser:

```bash
cd /Users/yuxinyu/handwriting-tool
source venv/bin/activate
venv/bin/python src/train_emnist_letters.py
```

Output model:

```text
models/emnist_letters_model.pth
```

**Typical output**

```text
Epoch 1/...
Epoch 2/...
Test Accuracy: ...
Model saved to .../models/emnist_letters_model.pth
```

### C. Single digit recognition

**Purpose**

Recognise one handwritten digit image.

```bash
venv/bin/python src/predict_image.py samples/raw/digits/digit_3_001_raw.png
```

**Typical output**

```text
Predicted digit: 3 (99.24%)
```

### D. Multi-digit recognition

**Purpose**

Segment a multi-digit handwritten image and recognise each digit in reading order.

```bash
venv/bin/python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png
```

**Typical output**

```text
Predicted digits:
1 2 3 4 5 6
7 8 9 0 3.1 4.2
5.3 7.8 6.9
```

Optional debug segment export:

```bash
venv/bin/python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png --save-segments samples/processed/digits/demo_multi_digit_segments
```

**Files produced**

```text
samples/processed/digits/demo_multi_digit_segments/
```

### E. Single letter recognition

**Purpose**

Recognise one handwritten letter image using the letter model.

```bash
venv/bin/python src/predict_image.py samples/raw/letters/letter_a_001_raw.png --label-set letters
```

**Typical output**

```text
Predicted letter: A (...)
```

### F. Sentence recognition

**Purpose**

Recognise a handwritten sentence image by segmenting character candidates, predicting letters, and applying English post-processing.

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
```

**Typical output**

```text
Predicted text:
Alice was beginning to get very tired of sitting by her
sister on the bank.
```

Optional debug segment export:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --save-segments samples/processed/letters/debug_sentence_alice
```

**Files produced**

```text
samples/processed/letters/debug_sentence_alice/
```

### G. Save a correction

**Purpose**

Run recognition and then ask the user for the correct label or text so the correction can be saved for future fine-tuning.

Single digit correction:

```bash
venv/bin/python src/predict_image.py samples/raw/digits/digit_3_001_raw.png --correct
```

**Typical output**

```text
Predicted digit: ...
Correct label/text [...]:
Correction saved to .../corrected_samples/labels.csv
```

Multi-digit correction:

```bash
venv/bin/python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png --correct
```

**Typical output**

```text
Predicted digits:
...
Correct label/text [...]:
Correction saved to .../corrected_samples/labels.csv
```

Sentence correction:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --correct
```

**What this specific command does**

This command:

1. recognises the handwritten sentence image
2. prints the predicted sentence
3. prompts the user to type the correct sentence
4. saves the correction
5. creates aligned letter-level correction samples for letter fine-tuning

**Typical output**

```text
Predicted text:
...
Correct label/text [...]:
Correction saved to .../corrected_samples/labels.csv
Saved ... aligned letter segment correction(s) to .../corrected_samples/letter_segments/labels.csv
```

Sentence corrections are saved in:

```text
corrected_samples/labels.csv
corrected_samples/letter_segments/labels.csv
```

**Files produced**

- `corrected_samples/labels.csv`
- `corrected_samples/images/`
- `corrected_samples/letter_segments/labels.csv`
- `corrected_samples/letter_segments/images/`

### H. Fine-tune the models

**Purpose**

Update the trained models using saved correction data and user handwriting samples.

Digit fine-tuning:

```bash
venv/bin/python src/fine_tune_corrections.py
```

**Typical output**

```text
Fine-tune epoch 1/...
Fine-tuned model saved to .../models/mnist_model.pth
Used ... correction sample(s)
```

Letter fine-tuning:

```bash
venv/bin/python src/fine_tune_letters.py
```

**Typical output**

```text
Fine-tune epoch 1/...
Fine-tuned model saved to .../models/emnist_letters_model.pth
Used ... aligned sentence segment sample(s)
Used ... font library letter sample(s)
```

### I. Personalised handwriting rendering

**Purpose**

Generate a handwriting-style output image from typed text or recognised text using the stored personal character libraries.

Render typed text directly:

```bash
venv/bin/python src/render_handwriting.py "Alice was beginning to get very tired of sitting by her sister on the bank." --output output/demo_direct_render.png
```

**Typical output**

```text
Rendered handwriting saved to .../output/demo_direct_render.png
```

Render recognised text directly from the recognition pipeline:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --render-recognized --render-output output/demo_rendered_from_recognition.png
```

**Typical output**

```text
Predicted text:
...
Rendered predicted text to output/demo_rendered_from_recognition.png
```

**Files produced**

- `output/demo_direct_render.png`
- `output/demo_rendered_from_recognition.png`

### J. TrueType font generation

**Purpose**

Vectorise the personalised character library (`samples/font_letters/` and `samples/font_digits/`) and export it as an installable `.ttf` font that can be used in any application.

```bash
venv/bin/python src/font_generator.py --output output/MyHandwriting.ttf --family-name "MyHandwriting"
```

**Typical output**

```text
Saved /Users/yuxinyu/handwriting-tool/output/MyHandwriting.ttf
```

**Files produced**

- `output/MyHandwriting.ttf`

To install, double-click the `.ttf` file in Finder and click "Install Font". The font will then be available in any application that supports custom fonts (Word, Pages, browsers, design tools, etc.).

## 5. OCR baseline comparison

**Purpose**

Use Tesseract as an external OCR baseline and compare its output with the custom pipeline.

Tesseract was used as an external OCR baseline.

Check installation:

```bash
tesseract --version
```

Run it on sentence images:

```bash
tesseract samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg stdout
tesseract samples/raw/letters/sentence_so_she_was_considering_001_raw.jpeg stdout
tesseract samples/raw/letters/sentence_but_it_had_no_001_raw.jpeg stdout
```

**Typical output**

```text
Alice@ was beg Inning ...
```

Compare these outputs with:

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
venv/bin/python src/predict_image.py samples/raw/letters/sentence_so_she_was_considering_001_raw.jpeg --label-set letters --multi
venv/bin/python src/predict_image.py samples/raw/letters/sentence_but_it_had_no_001_raw.jpeg --label-set letters --multi
```

**Result location**

See the qualitative comparison summary in:

- [evaluation_summary.md](/Users/yuxinyu/handwriting-tool/proposal/evaluation_summary.md)

## 6. Best files to show during the presentation

### Best image outputs

- [output/rendered_from_recognition_sentence_case.png](/Users/yuxinyu/handwriting-tool/output/rendered_from_recognition_sentence_case.png)
- [output/rendered_from_recognition_alice.png](/Users/yuxinyu/handwriting-tool/output/rendered_from_recognition_alice.png)
- [output/rendered_Hello-World.png](/Users/yuxinyu/handwriting-tool/output/rendered_Hello-World.png)
- [output/rendered_314159.png](/Users/yuxinyu/handwriting-tool/output/rendered_314159.png)
- [output/MyHandwriting.ttf](/Users/yuxinyu/handwriting-tool/output/MyHandwriting.ttf) (installable font — open in Font Book to preview, then install and type with it in any application)

### Best terminal demonstrations

- single digit recognition
- multi-digit recognition
- Alice sentence recognition
- correction and fine-tuning loop
- recognise-then-render workflow
- TTF font generation and installation

## 7. Best live-demo order

Recommended sentence demo:

1. recognise a handwritten sentence
2. correct the sentence
3. fine-tune the letter model
4. recognise again
5. render recognised text back into handwriting
6. generate a `.ttf` from the user's character library and install it

See:

- [demo_script.md](/Users/yuxinyu/handwriting-tool/proposal/demo_script.md)
- [demo_commands.md](/Users/yuxinyu/handwriting-tool/proposal/demo_commands.md)
- [demo_presentation_notes.md](/Users/yuxinyu/handwriting-tool/proposal/demo_presentation_notes.md)
