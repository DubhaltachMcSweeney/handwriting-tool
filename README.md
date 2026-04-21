# handwriting-tool

aka Handwriting Digitisation and Personalized Font Generation Tool

## Project Guide

This repository currently implements a prototype for handwritten digit recognition. It can train a simple MNIST model, preprocess single digit images, segment multi-digit images, run predictions from the command line, and store user corrections for later retraining.

At this stage, the model is trained on MNIST and is intended for digits only (`0-9`). It does not reliably recognise letters yet. Letter recognition would require a separate dataset such as EMNIST or an OCR tool.

### Main Files

- `src/train_mnist.py`: trains a simple neural network on the MNIST digit dataset and saves the model to `models/mnist_model.pth`.
- `src/predict_image.py`: command-line prediction tool for single digit images and multi-digit images. It can also save corrected labels.
- `src/preprocessing.py`: image preprocessing utilities. It handles grayscale conversion, transparent image backgrounds, binarisation, content cropping, resizing, centering, and tensor conversion.
- `src/segmentation.py`: OpenCV-based segmentation for multi-digit images. It finds digit-like contours, groups them by line, preserves decimal points when appropriate, and outputs `28x28` segments.
- `src/corrections.py`: saves user correction records to `corrected_samples/labels.csv` and copies corrected images into `corrected_samples/images/`.
- `src/image_loader.py`: simple OpenCV image loading demo used for early image input testing.
- `tests/`: unit tests for preprocessing, segmentation, and correction saving.
- `samples/`: example handwritten images and sample naming instructions.
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
/Users/yuxinyu/handwriting-tool/venv/bin/python
```

### Train the Model

Run:

```bash
python src/train_mnist.py
```

This downloads MNIST into `data/`, trains the model, evaluates it on the MNIST test set, and saves:

```text
models/mnist_model.pth
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

### Run Tests

Run:

```bash
python -m pytest
```

The tests cover missing image files, invalid image files, different image sizes, white-background and black-background images, single digit preprocessing, multi-digit segmentation, decimal point handling, and correction saving.

### Sample Naming

Use clear names for new sample images:

```text
samples/raw/digits/digit_<label>_<number>_raw.png
samples/raw/digits/digit_unknown_<number>_raw.png
samples/raw/multi_digit/multi_digit_<text>_<number>_raw.png
samples/raw/letters/letter_<label>_<number>_raw.png
```

Examples:

```text
samples/raw/digits/digit_5_001_raw.png
samples/raw/digits/digit_unknown_001_raw.png
samples/raw/multi_digit/multi_digit_123_001_raw.png
samples/raw/letters/letter_a_001_raw.png
```

Use `unknown` when the true label has not been confirmed. Do not name files only by the model prediction unless the image has been checked manually.

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
