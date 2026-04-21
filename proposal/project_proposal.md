# Handwriting Digitisation and Personalized Font Generation Tool

## 1. Mission

The mission of this project is to design and implement a tool that can convert a user’s handwritten characters into a digital representation and, where possible, generate personalized font-like output based on the user’s handwriting style. The system aims to bridge handwritten input and digital text by combining image preprocessing, character recognition, and interactive user feedback.

This project also explores the role of machine learning in handwriting analysis. In particular, the system may use a neural network to classify handwritten characters and improve its accuracy through an interactive correction loop, where users can confirm or correct predictions. The overall goal is to create an intuitive prototype that demonstrates both practical usability and scientific validity in handwriting recognition and digitisation.

## 2. Scope

### In Scope

- Uploading images of handwritten characters or words.
- Preprocessing handwriting images (e.g. grayscale conversion, resizing, thresholding, segmentation).
- Recognising handwritten characters using either:
  - a simple OCR-based approach (e.g. pretrained tools such as Tesseract), or
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
- Compare the performance of pretrained OCR tools (e.g. Tesseract) with custom-trained machine learning models.

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