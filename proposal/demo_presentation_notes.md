# Demo Presentation Notes

This file is an English speaking script for presenting the `sentence_alice_was_beginning_001_raw.jpeg` demo.

## Opening

```text
Our project is a handwriting digitisation and personalised handwriting rendering tool.

Instead of only recognising handwriting once, we built an interactive pipeline:
the system recognises handwriting, the user can correct mistakes, the model can be fine-tuned on those corrections, and then the recognised text can be rendered back into the user’s own handwriting style.
```

## Step 1. Show the handwritten sentence image

```text
Here we start with a handwritten sentence image written by the user.
This is the raw input to the system.
```

## Step 2. First recognition pass

```text
First, we run the recognition pipeline.

The image is preprocessed, segmented into character candidates, passed through the letter recognition model, and then repaired with dictionary-based English post-processing.
```

## Step 3. Explain the correction step

```text
If the recognition is not fully correct, the user can provide the correct sentence.

We save both the whole correction and aligned character-level correction samples.
This is important because it turns user feedback into new training data.
```

## Step 4. Fine-tuning

```text
Next, we fine-tune the letter model on the corrected user handwriting.

This makes the recogniser more adapted to the user’s own writing style, rather than only depending on the original EMNIST training data.
```

## Step 5. Second recognition pass

```text
After fine-tuning, we run recognition again on the same sentence.

The goal here is to show that the system can improve through interaction and user-specific adaptation.
```

## Step 6. Personalised handwriting rendering

```text
Finally, we take the recognised text and render it back into a handwriting-style image using the user’s stored character library.

This means the system does not stop at recognition. It also supports a simple personalised handwriting generation workflow.
```

## Why this matters

```text
What makes this project meaningful is that it is not just a static classifier.

It connects:
recognition,
user correction,
model improvement,
and personalised output generation.

So the project demonstrates both practical usability and an interactive machine learning workflow.
```

## Honest limitation statement

```text
Digit recognition is currently the most reliable part of the project.

Sentence recognition still struggles with connected lowercase handwriting, spacing ambiguity, and long natural text.

However, the correction-and-fine-tuning loop clearly improves the system and shows the value of a personalised approach.
```

## Closing

```text
In summary, our project shows a complete prototype:
handwritten input,
digital text output,
interactive correction,
user-specific learning,
and personalised handwriting-style rendering.
```
