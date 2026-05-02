# Evaluation Summary

This document summarises representative results from the current prototype.

It is intended as a compact evaluation note for the proposal, report, or presentation. The focus is on:

- digit recognition quality
- sentence recognition improvement through correction and fine-tuning
- personalised handwriting rendering output

## 1. Evaluation Setup

The project currently combines:

- a digit model trained on MNIST and later adapted with user-provided digit samples
- a letter model trained on EMNIST Letters and later adapted with:
  - aligned sentence-level letter corrections
  - raw labelled letter samples
  - the personal `font_letters` character library
- OpenCV-based preprocessing and segmentation
- dictionary-assisted English post-processing for sentence repair

The results below mix:

1. **representative development checkpoints** recorded while improving the pipeline
2. **latest spot-check outputs** from the current codebase

This is useful because the project evolved from a basic recogniser into a more complete interactive prototype.

## 2. Digits

| Task | Sample | Earlier / Before | Current / After | Interpretation |
|---|---|---|---|---|
| Single digit recognition | `samples/raw/digits/digit_3_001_raw.png` | `Predicted digit: 3 (44.33%) [LOW CONFIDENCE]` | `Predicted digit: 3 (99.24%)` | Confidence improved substantially after model upgrades and user-style adaptation. |
| Multi-digit recognition | `samples/raw/multi_digit/multi_digit_unknown_001_raw.png` | `12341618175.14.25.17.86.1` | `1 2 3 4 5 6` / `7 8 9 0 3.1 4.2` / `5.3 7.8 6.9` | The pipeline moved from a mostly unusable result to a readable and nearly correct structured output. |

### Digit Summary

- Digit recognition is currently the strongest part of the project.
- The biggest improvements came from:
  - better preprocessing
  - better segmentation
  - replacing the simple model with a CNN
  - fine-tuning with user handwriting samples

## 3. Letters and Sentences

### Alice Sentence Progression

Ground-truth sentence:

```text
Alice was beginning to get very tired of sitting by her sister on the bank.
```

| Stage | Output |
|---|---|
| Before correction / early sentence prototype | `HLIOB MAS WEYINHING W GET VQRY TGNPD Y SJTTANG MY HER` / `GJSTOI AN QHE BAWK.` |
| After letter fine-tuning | `ALICE WAS BEGINNING W GET VORY TINED Y SITTING NY HER` / `SISTER AN THE BANK.` |
| After English post-processing and sentence-case restoration (representative best run during development) | `Alice was beginning to get very tired of sitting by her sister on the bank.` |

### Latest Spot-Check on the Same Sentence

Current output from the latest code snapshot:

```text
Alice was beginning to get very tired of sitting by her
so sister on the bank.
```

This shows two important things:

1. the system can now produce readable sentence-level output
2. sentence recognition is still the least stable part of the project and remains sensitive to segmentation and post-processing changes

### Additional Sentence Samples

The project was further tested on longer handwritten English sentences:

- `sentence_whether_the_pleasure_001_raw.jpeg`
- `sentence_so_she_was_considering_001_raw.jpeg`
- `sentence_but_it_had_no_001_raw.jpeg`

Representative current outputs show partial success:

- common words are often recovered correctly
- long or connected words still degrade
- dictionary repair helps, but does not fully solve sentence OCR

Examples from current spot-checks:

```text
Whether the pleasure of making had daisy chain would be would be worth the trouble
getting up and picking the daisies.
```

```text
So she was considering in her own mind las well as she could oihr the hot day made
hot feel very sleepy srupi do
```

```text
But it had no pictures or conversations in it I in own nhhot is the
use of a book. Thought alice without pictures or an msatiohsn.
```

### Letter / Sentence Summary

- Sentence recognition has clearly improved from its earliest stage.
- The correction + fine-tuning loop is working and materially improves user-specific handwriting recognition.
- However, sentence-level performance is still limited by:
  - connected lowercase handwriting
  - character merging and spacing ambiguity
  - the lack of a stronger sequence model

## 4. Personalised Output: Rendering and Font Generation

The project now supports a simple personalised handwriting rendering prototype based on reusable character libraries:

- `samples/font_digits/`
- `samples/font_letters/uppercase/`
- `samples/font_letters/lowercase/`

### 4.1 Handwriting-Style PNG Rendering

The renderer composes typed text into a handwriting-style image by stitching individual character samples onto a canvas, with optional baseline jitter and random sample selection for natural variation.

### Representative Rendering Examples

| Input text | Output file | Notes |
|---|---|---|
| `314159` | [output/rendered_314159.png](/Users/yuxinyu/handwriting-tool/output/rendered_314159.png) | Personal digit rendering from the digit font library |
| `Hello-World` | [output/rendered_Hello-World.png](/Users/yuxinyu/handwriting-tool/output/rendered_Hello-World.png) | Letter rendering with punctuation support |
| recognised Alice sentence | [output/rendered_from_recognition_sentence_case.png](/Users/yuxinyu/handwriting-tool/output/rendered_from_recognition_sentence_case.png) | End-to-end recognition-to-rendering demonstration |
| recognised “and of having...” sentence | [output/rendered_and_of_having_font_letters.png](/Users/yuxinyu/handwriting-tool/output/rendered_and_of_having_font_letters.png) | Personal sentence rendering after recognition |

#### Rendering Summary

- The renderer successfully demonstrates the "personalised handwriting-style output" goal from the proposal.
- This part is stronger as a prototype than the sentence OCR stage because it relies on the stored character library rather than uncertain sequence recognition.
- The output is a one-off PNG image — useful for visual demonstration, but limited in further use.

### 4.2 TrueType Font (.ttf) Generation

The project also exports the personalised character library as an installable TrueType font, generated by vectorising each glyph PNG with Potrace and assembling the curves into a font file with fontTools.

This produces a `.ttf` file that can be installed on any operating system and used in any application that supports custom fonts (Word, Pages, browsers, design tools, etc.), turning the user's handwriting into a reusable typeface rather than a single rendered image.

#### Generation Pipeline

```text
samples/font_letters///letter__001.png
        |
        v
binarise -> Potrace trace -> fontTools glyph
        |
        v
assemble all 62 glyphs (26 uppercase + 26 lowercase + 10 digits)
        |
        v
output/MyHandwriting.ttf
```

#### Output

| Output file | Description |
|---|---|
| `output/MyHandwriting.ttf` | Installable TrueType font containing 62 glyphs vectorised from the user's handwriting samples |

#### Font Generation Summary

- All 62 target glyphs are produced with no failures during vectorisation.
- The resulting font installs and renders correctly in standard applications including TextEdit, Word, and browsers.
- Glyph shapes are recognisably faithful to the source handwriting samples.
- The output is functional but not typographically polished. Known limitations:
  - Inconsistent baseline alignment between glyphs (each glyph uses the bounding box of its source PNG, with no shared baseline detection)
  - Inconsistent x-height for lowercase letters
  - No kerning, ligatures, or other OpenType features (out of scope by the proposal)
- Compared with the PNG renderer, the TTF output is significantly more useful in practice: it produces real text in any application rather than a single rasterised image.

## 5. Proposal Alignment

### Clearly Achieved

- image preprocessing
- digit recognition
- letter recognition in a constrained setting
- user correction loop
- reusable handwriting character library
- personalised handwriting-style rendering
- installable TrueType (.ttf) font export from the personalised character library
- unit testing for core image-processing and correction workflows

### Partially Achieved

- sentence-level handwritten English recognition
- generalisation to longer unseen sentence samples
- robust word/space segmentation for continuous lowercase writing

### Not Yet Fully Addressed

- a formal OCR baseline comparison (for example, Tesseract vs. custom pipeline)
- a stronger quantitative accuracy table across multiple handwritten sentence samples

## 6. Main Takeaway

The most meaningful result of this project is not just that it recognises handwriting, but that it demonstrates a full interactive loop:

```text
handwritten image -> segmentation -> recognition -> user correction -> fine-tuning -> improved recognition -> personalised rendering / installable TTF font
```

That makes the project more valuable than a simple static classifier. It shows:

- practical handwriting digitisation
- user-in-the-loop machine learning
- adaptation to personal handwriting style
- a bridge from recognition into personalised output generation

## 7. Recommended Next Additions for the Final Report

To make the final submission stronger, the next most useful additions are:

1. add a small OCR baseline comparison
2. turn this summary into one clean table in the report
3. include before/after screenshots for:
   - multi-digit recognition
   - Alice sentence recognition
   - personalised rendering output
   - generated TTF font samples (text typed in the font, before/after install)

## 8. OCR Baseline Comparison (Tesseract)

To add a lightweight external baseline, the project was compared against **Tesseract 5.5.2** on three handwritten sentence images.

This comparison is qualitative rather than fully statistical, but it is still useful for the proposal because it shows how a pretrained OCR tool behaves on the same handwritten inputs as the custom pipeline.

### Comparison Table

| Image | Ground Truth | Custom Pipeline | Tesseract | Notes |
|---|---|---|---|---|
| `sentence_alice_was_beginning_001_raw.jpeg` | `Alice was beginning to get very tired of sitting by her sister on the bank.` | `Alice was beginning to get very tired of sitting by her / so sister on the bank.` | `Alice@ was beg Inning wo Jet very tired oy Sitting by her / Cister on the bone` | The custom pipeline preserves sentence structure and more words correctly. Tesseract gets some words right, but degrades strongly on `sister` and `bank`. |
| `sentence_so_she_was_considering_001_raw.jpeg` | `So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid)` | `So she was considering in her own mind las well as she could oihr the hot day made / hot feel very sleepy srupi do` | `Go She Was Considering Tn h@r pwh mind (a5 will @4 She Could ,for the hot doy made / her fee| very sleepy and Stupid,` | Both systems struggle, but each captures parts of the sentence. The custom system recovers the sentence beginning more naturally; Tesseract preserves `sleepy and stupid` better here. |
| `sentence_but_it_had_no_001_raw.jpeg` | `but it had no pictures or conversations in it, and what is the use of a book, thought Alice without pictures or conversations?` | `But it had no pictures or conversations in it I in own nhhot is the / use of a book. Thought alice without pictures or an msatiohsn.` | `but It had ho picwres or Conversations Tn it," ond Wwhot 14 the / Woe of a book ,” thought Alice without pictues or Conversations?` | Both systems recover several high-level words. The custom system is stronger on `pictures or conversations` in the first clause, while Tesseract is stronger on punctuation and the `thought Alice` portion. |

### Baseline Takeaway

The OCR baseline is useful because it shows that the custom pipeline is not simply reproducing what an off-the-shelf tool already does.

Main observations:

- **Tesseract is competitive on some isolated words and punctuation-heavy parts.**
- **The custom pipeline benefits from user correction, handwriting-specific fine-tuning, and personalised rendering.**
- **Neither system is fully reliable on long connected lowercase handwriting.**
- **The custom system is more aligned with the project goal** because it supports:
  - interactive correction
  - user-specific learning
  - reusable character libraries
  - recognition-to-rendering output generation

This makes the baseline comparison valuable for the final report even if the comparison remains qualitative.
