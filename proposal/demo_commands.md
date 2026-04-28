# Demo Command Checklist

Use this file when you want the shortest possible live-demo command list.

## Setup

```bash
cd /Users/yuxinyu/handwriting-tool
source venv/bin/activate
```

## 1. Initial sentence recognition

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
```

## 2. Correction step

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --correct
```

Correction text to paste:

```text
Alice was beginning to get very tired of sitting by her sister on the bank.
```

## 3. Fine-tune after correction

```bash
venv/bin/python src/fine_tune_letters.py
```

## 4. Recognise again after fine-tuning

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi
```

## 5. Render recognised text into handwriting

```bash
venv/bin/python src/predict_image.py samples/raw/letters/sentence_alice_was_beginning_001_raw.jpeg --label-set letters --multi --render-recognized --render-output output/demo_rendered_sentence.png
```

## 6. Optional digit demonstration

```bash
venv/bin/python src/predict_image.py --multi samples/raw/multi_digit/multi_digit_unknown_001_raw.png
```

## 7. Optional direct text rendering

```bash
venv/bin/python src/render_handwriting.py "Alice was beginning to get very tired of sitting by her sister on the bank." --output output/demo_direct_render.png
```
