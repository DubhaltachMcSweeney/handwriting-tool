# handwriting-tool

aka Handwriting Digitisation and Personalized Font Generation Tool

## Project Guide

This repository currently implements a handwriting digitisation and personalised handwriting prototype rather than only a digit demo.

It now supports:

- handwritten digit recognition
- handwritten letter recognition
- simple sentence-level handwritten English recognition
- user correction and feedback-driven fine-tuning
- personalised handwriting-style rendering from stored character libraries

The current system combines:

- a digit model trained on MNIST and adapted with user handwriting samples
- a letter model trained on EMNIST Letters and adapted with user sentence and character samples
- OpenCV-based preprocessing and segmentation
- English dictionary post-processing for recognised sentence text
- reusable handwriting character libraries for digits and letters

This is still a constrained prototype, not a production OCR system. Digit recognition is currently the most reliable part. Letter and sentence recognition work as a prototype, but still depend heavily on image quality, spacing, and writing style.

### Current Status

The project currently supports four connected workflows:

1. **Digit recognition**
   - single-digit prediction
   - multi-digit prediction
   - decimal-aware segmentation

2. **Letter and sentence recognition**
   - single handwritten letter recognition
   - handwritten sentence recognition prototype
   - English post-processing and sentence-case restoration

3. **Interactive improvement**
   - saving corrections
   - aligning corrected sentence text to segmented character images
   - fine-tuning models on user-specific handwriting

4. **Personalised handwriting rendering**
   - rendering typed text from `font_digits` and `font_letters`
   - rendering recognised text back into a handwriting-style image

### Main Files

- `src/train_mnist.py`: trains the digit model and saves it to `models/mnist_model.pth`.
- `src/train_emnist_letters.py`: trains the letter model and saves it to `models/emnist_letters_model.pth`.
- `src/fine_tune_corrections.py`: fine-tunes the digit model using saved correction samples and labelled multi-digit samples.
- `src/fine_tune_letters.py`: fine-tunes the letter model using aligned sentence corrections, raw letter samples, and the personal font-letter library.
- `src/predict_image.py`: the main command-line prediction tool for digits, letters, multi-digit images, and sentence images. It also supports correction recording and rendering recognised text back into handwriting.
- `src/preprocessing.py`: shared image preprocessing utilities for grayscale conversion, binarisation, cropping, resizing, centering, and tensor conversion.
- `src/segmentation.py`: OpenCV-based segmentation for multi-digit images with decimal-point handling.
- `src/text_segmentation.py`: sentence/text segmentation for letter and sentence images, including line grouping and punctuation handling.
- `src/english_postprocess.py`: dictionary-based English word and sentence repair for recognised letter text.
- `src/corrections.py`: stores general correction records in `corrected_samples/labels.csv`.
- `src/letter_corrections.py`: stores aligned sentence-to-character correction records for letter fine-tuning.
- `src/render_handwriting.py`: renders typed or recognised text back into a handwriting-style image using the character libraries.
- `src/model.py`: shared CNN model definitions for digit and letter recognition.
- `src/recognition_config.py`: label-set and model-path configuration helpers.
- `src/image_loader.py`: early image loading demo kept for simple manual inspection.
- `tests/`: unit tests for preprocessing, segmentation, corrections, post-processing, fine-tuning sample loading, and rendering.
- `samples/`: handwritten input samples, processed debug outputs, font libraries, and naming instructions.
- `proposal/project_proposal.md`: project proposal and planning document.

### Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the interpreter:

```text
Python: Select Interpreter
/Your path/handwriting-tool/venv/bin/python
```

### Train the Models

Run:

```bash
python src/train_mnist.py
```

This downloads MNIST into `data/`, trains the digit model, evaluates it on the MNIST test set, and saves:

```text
models/mnist_model.pth
```

For letters, run:

```bash
python src/train_emnist_letters.py
```

This trains the letter model on EMNIST Letters and saves:

```text
models/emnist_letters_model.pth
```

### Predict a Single Digit

Run:

```bash
python src/predict_image.py samples/raw/digits/digit_3_001_raw.png
```

Example output:

```text
Predicted digit: 3
```

### Predict a Single Letter

Run:

```bash
python src/predict_image.py samples/raw/letters/letter_a_001_raw.png --label-set letters
```

Example output:

```text
Predicted letter: A
```

### Predict Multiple Digits

For an image containing multiple digits, run:

```bash
python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png
```

This segments the image and predicts each detected digit from top to bottom and left to right.

To save the segmented `28x28` images for debugging:

```bash
python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png --save-segments samples/processed/digits/debug_unknown_001
```

### Predict a Handwritten Sentence

For a handwritten sentence image, run:

```bash
python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
```

This will:

- segment the sentence into character candidates
- classify them with the letter model
- post-process the result with English dictionary repair and sentence-case restoration

### Save Debug Segments for Sentence Recognition

To save segmented character images for debugging:

```bash
python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --save-segments samples/processed/letters/debug_sentence_alice
```

### Save a User Correction

For a single digit:

```bash
python src/predict_image.py samples/raw/digits/digit_3_001_raw.png --correct
```

For multiple digits:

```bash
python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png --correct
```

The program will ask for the correct label or text. Corrections are saved in:

```text
corrected_samples/labels.csv
corrected_samples/images/
```

For sentence-level letter correction, the system also saves aligned character correction samples in:

```text
corrected_samples/letter_segments/labels.csv
corrected_samples/letter_segments/images/
```

### Fine-Tune the Models

To fine-tune the digit model with saved corrections:

```bash
python src/fine_tune_corrections.py
```

To fine-tune the letter model with sentence corrections, raw letter samples, and personal font-letter samples:

```bash
python src/fine_tune_letters.py
```

### Render Personalised Handwriting

To render a text string using the digit/letter character libraries:

```bash
python src/render_handwriting.py "Alice was beginning" --output output/rendered_alice.png
```

This uses:

- `samples/font_digits/`
- `samples/font_letters/uppercase/`
- `samples/font_letters/lowercase/`

### Recognise and Immediately Re-Render

To recognise a handwritten sentence and immediately convert the recognised text back into your stored handwriting style:

```bash
python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --render-recognized --render-output output/rendered_from_recognition.png
```

### Run Tests

Run:

```bash
python -m pytest
```

The tests cover:

- missing and invalid image files
- different image sizes and image polarity
- digit preprocessing
- multi-digit segmentation and decimal handling
- correction saving
- sentence-segment correction loading
- text post-processing
- rendering helper behavior

### Sample Naming

Use clear names for new sample images:

```text
samples/raw/digits/digit_<label>_<number>_raw.png
samples/raw/digits/digit_unknown_<number>_raw.png
samples/raw/multi_digit/multi_digit_<text>_<number>_raw.png
samples/raw/letters/letter_<label>_<number>_raw.png
samples/raw/letters/sentence_<description>_<number>_raw.jpeg
```

Examples:

```text
samples/raw/digits/digit_5_001_raw.png
samples/raw/digits/digit_unknown_001_raw.png
samples/raw/multi_digit/multi_digit_123_001_raw.png
samples/raw/letters/letter_a_001_raw.png
samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg
```

Use `unknown` when the true label has not been confirmed. Do not name files only by the model prediction unless the image has been checked manually.

### Current Limitations

- Sentence recognition is still a prototype and can struggle with connected lowercase writing.
- The letter model is more reliable on constrained samples than on long natural sentences.
- English post-processing improves readability, but it does not replace strong sequence modelling.
- The renderer is a character-library prototype, not a full font engine.

## 1. Mission

The mission of this project is to design and implement a tool that can convert a user’s handwritten characters into a digital representation and, where possible, generate personalized font-like output based on the user’s handwriting style. The system aims to bridge handwritten input and digital text by combining image preprocessing, character recognition, and interactive user feedback.

This project also explores the role of machine learning in handwriting analysis. In particular, the system may use a neural network to classify handwritten characters and improve its accuracy through an interactive correction loop, where users can confirm or correct predictions. The overall goal is to create an intuitive prototype that demonstrates both practical usability and scientific validity in handwriting recognition and digitisation.

## 2. Scope

### In Scope

- Uploading images of handwritten characters or words.
- Preprocessing handwriting images (e.g. grayscale conversion, resizing, thresholding, segmentation).
- Recognising handwritten characters using either:
  - a simple OCR-based approach, or
  - a neural network / machine learning model.
- Allowing users to label or correct characters manually.
- Building an interactive feedback loop where corrections can be used to improve future recognition.
- Converting recognised characters into digital text.
- Exploring a simple method to generate a personalized handwriting-style font or reusable character set.
- Testing the system on known datasets and new user-provided samples.
- Writing unit tests for different image sizes, formats, and basic processing functions.

### Out of Scope / Non-Goals

- Building a production-quality commercial font generation platform.
- Supporting all world languages or highly complex writing systems.
- Real-time recognition for large-scale scanned documents.
- Fully automatic font generation with no user correction.
- Advanced typography features such as kerning, ligatures, or professional font design standards.


## 3. Objectives

### Scientific Validity Objectives

- Investigate whether machine learning can reliably recognise user handwriting in a constrained setting.
- Compare recognition performance on:
  - known training-style data,
  - unseen handwriting samples,
  - noisy or “gibberish” inputs.
- Evaluate how preprocessing affects recognition accuracy.
- Measure the benefit of user correction in an interactive feedback loop.
- Demonstrate that the system can generalise beyond a fixed dataset such as MNIST, while acknowledging limitations.

### Operational Performance Objectives

- Create a working prototype that allows image upload and handwriting analysis.
- Ensure the system accepts common image formats and handles different image sizes robustly.
- Produce readable digital text output from handwritten input.
- Provide a user-friendly correction interface for mislabeled characters.
- Keep processing time reasonable for small handwriting samples.
- Demonstrate stable performance through testing and debugging.

## 4. Inputs / Outputs

### Inputs

- Images containing handwritten characters, digits, or words.
- User-provided labels for characters during correction or manual training.
- Optional benchmark datasets (e.g. MNIST or EMNIST) for initial model testing.
- Configuration settings for preprocessing steps such as image size, thresholding, or grayscale conversion.

### Outputs

- Predicted digital text corresponding to the handwritten input.
- A segmented and processed representation of the handwritten characters.
- User-corrected labels for retraining or refinement.
- A simple personalized font prototype, character library, or rendered text output that imitates the user’s handwriting style.
- Accuracy metrics, test results, and system evaluation reports.

## 5. Constraints

- Limited project time and team resources may restrict model complexity and feature completeness.
- Handwriting recognition accuracy may vary significantly across users and writing styles.
- Font generation is technically complex and may only be feasible as a simplified prototype.
- Training a neural network requires sufficient data and careful preprocessing.
- The team may need to balance between using existing libraries and implementing custom methods.
- Performance may depend heavily on image quality, lighting conditions, and input consistency.
- Some team members may need additional research time to understand OCR, image processing, and neural network design.

## 6. Risks & Mitigation Strategies

- Risk: Poor handwriting recognition accuracy.  
  Mitigation:Start with a constrained character set (e.g. digits or uppercase letters), use preprocessing carefully, and include manual correction.

-- Risk: Input images vary too much in quality and format.  
  Mitigation: Define accepted input requirements and implement preprocessing plus unit tests for multiple image sizes and formats.
