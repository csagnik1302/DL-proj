# Bangla Literary Style Transfer

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu130-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.2-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![pdfplumber](https://img.shields.io/badge/pdfplumber-0.11.9-4B8BBE)](https://github.com/jsvine/pdfplumber)
[![pytesseract](https://img.shields.io/badge/pytesseract-0.3.13-5C2D91)](https://github.com/madmaze/pytesseract)

An experimental deep-learning system for transferring Bangla sentences between the literary styles of five canonical Bengali authors. The project learns a style-invariant content representation and uses a target-author embedding to generate a rewritten sentence.

> **Research prototype:** Generated text is probabilistic and may contain factual, grammatical, or stylistic errors. It should not be presented as authentic writing by an author.

## Highlights

- Supports five author styles: Bibhutibhushan Bandopadhyay, Rabindranath Tagore, Sarat Chandra Chattopadhyay, Satyajit Ray, and Sunil Gangopadhay.
- Builds a sentence-level Bangla corpus from literary PDFs, using `pdfplumber` with a Tesseract OCR fallback.
- Filters common OCR noise and reports corpus-quality statistics.
- Uses adversarial representation learning to separate sentence content from author identity.
- Uses SentencePiece subword tokenization consistently across training and inference in the current experiment.
- Includes interactive inference with temperature sampling, top-k filtering, and a repetition penalty.

## Model Overview

The architecture follows a domain-adversarial style-transfer approach:

1. A 128-dimensional word embedding layer encodes token IDs.
2. A bidirectional GRU maps a sentence to a 260-dimensional latent representation.
3. An author discriminator predicts the source author from this representation.
4. A Gradient Reversal Layer trains the encoder to make author prediction difficult, encouraging a style-invariant representation.
5. A unidirectional GRU decoder receives the encoded content, its previous token, and a learned target-author embedding to generate the output sentence.

The training objective combines reconstruction loss with an adversarial author-classification loss. The adversarial signal is introduced after a warm-up period to preserve reconstruction quality early in training.

### Tokenization Evolution

The initial model used word-level tokenization, with words occurring five or fewer times mapped to `<unk>`. The current update replaces this with [SentencePiece](https://github.com/google/sentencepiece) subword tokenization across both training and inference, eliminating training/inference vocabulary mismatch and improving coverage of rare or morphologically complex Bangla words.

The active tokenizer is implemented in `model/vocab_creator.py`. The former word-level tokenization and encoding functions remain commented out in `model/vocab_creator.py`, `model/input_creator.py`, and `model/inference.py` as a fallback for future baseline experiments.

## Dataset

The released corpus contains **18,779 sentence-author pairs** from five Bengali authors. Sentences have at least five words and pass Bangla-specific OCR and linguistic-quality filters.

| Author | Sentences | Mean OCR noise score |
| --- | ---: | ---: |
| Bibhutibhushan Bandopadhyay | 4,500 | 0.116 |
| Rabindranath Tagore | 3,279 | 0.064 |
| Sarat Chandra Chattopadhyay | 3,500 | 0.099 |
| Satyajit Ray | 4,000 | 0.095 |
| Sunil Gangopadhay | 3,500 | 0.084 |

The dataset is publicly available as [`sentences.csv`](https://github.com/csagnik1302/DL-proj/blob/main/Dataset_Creator/bangla_corpus/sentences.csv). It is a sentence-level CSV with `author` and `text` columns.

## Reported Training Results

### Word-Level Baseline

In the original 200-epoch word-level experiment, discriminator accuracy rose to approximately **65%** during the first 30 warm-up epochs, then declined toward the **20% random-chance baseline** after gradient reversal was enabled. This behaviour indicates that the encoder learned to reduce author-identifying information in its latent representation.

The reported baseline uses Adam with a learning rate of `1e-4` for both model components, a batch size of 64, a 30-epoch warm-up, and a maximum gradient-reversal strength of `0.0001`. Model selection uses the checkpoint with the lowest reconstruction loss.

### SentencePiece Update

The subword experiment was trained for 300 epochs. Its discriminator accuracy peaked at approximately **47%**, below the word-level baseline, and did not return to the **20%** chance-level floor. This indicates that author-style information remains in the GRU latent representation and the adversarial schedule requires further tuning for the new vocabulary.

Qualitative evaluation nevertheless found that SentencePiece captures local Bangla subword dependencies better than the word-level baseline. Output fluency is marginally improved, though repetition still occurs and the current temperature and repetition-penalty settings require tuning for subword generation.

## Repository Structure

```text
.
├── Dataset_Creator/
│   ├── bangla_corpus_builder.py  # Extract and clean sentences from PDFs
│   ├── corpus_validator.py       # Print corpus and OCR-quality statistics
│   └── requirements.txt          # Corpus-building dependencies
├── model/
│   ├── Model.py                  # Training entry point
│   ├── Adversary_Passes.py       # Adversarial encoder/decoder/discriminator passes
│   ├── input_creator.py          # Token and author tensor preparation
│   ├── vocab_creator.py          # Tokenisation and vocabulary construction
│   ├── inference.py              # Interactive style-transfer CLI
│   └── weights/                  # Trained checkpoint and vocabulary
├── Project_Report.pdf            # Full project report
└── Project_Slides.pdf            # Presentation slides
```

## Requirements

- Python 3.10 or newer
- [PyTorch](https://pytorch.org/) (CPU or CUDA build appropriate for your machine)
- Tesseract OCR with the Bengali (`ben`) language data, for scanned PDFs
- Poppler, for PDF-to-image conversion during OCR fallback

Create and activate a virtual environment, then install the listed corpus dependencies plus the model runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Dataset_Creator\requirements.txt
pip install torch matplotlib sentencepiece
```

Set the executable paths near the top of `Dataset_Creator/bangla_corpus_builder.py` (or `Dataset_Creator/corpus_validator.py`) to match your Tesseract and Poppler installations. The defaults are Windows paths and will need changing on another machine.

## Build a Corpus

Organise source PDFs by author. Folder names become the author labels, so use the expected names when training this model:

```text
source_pdfs/
├── bibhutibhushan bandopadhyay/
│   └── book-one.pdf
├── rabindranath tagore/
│   └── book-one.pdf
├── sarat chandra chattopadhyay/
├── satyajit ray/
└── sunil gangopadhay/
```

Run the corpus builder from the repository root:

```powershell
python Dataset_Creator\bangla_corpus_builder.py .\source_pdfs
```

It creates `Dataset_Creator/bangla_corpus/sentences.csv` with `author` and `text` columns. The script first attempts embedded-text extraction and falls back to Bengali OCR when necessary. It also records processed files to avoid repeated work.

Validate the resulting dataset before training:

```powershell
python Dataset_Creator\corpus_validator.py Dataset_Creator\bangla_corpus\sentences.csv
```

## Train

`model/Model.py` is the training entry point. Before running it, update the `pd.read_csv(...)` path in that file to point to your local `sentences.csv`; it currently contains a machine-specific absolute path.

Then run it from the `model` directory so its local imports resolve correctly:

```powershell
cd model
python Model.py
```

The default configuration uses a batch size of 64, up to 400 epochs, and automatically selects CUDA when available. Checkpoints and training plots are produced by the training code.

Training creates `model/bangla_sentencepiece.model`. Keep this file together with the matching `weights.pth` and `vocab.pth`; all three are required for SentencePiece inference.

## Run Interactive Inference

Trained assets are stored in `model/weights/`. Before the first run, update the two constants at the top of `model/inference.py` to:

```python
WEIGHTS_PATH = "weights/weights.pth"
VOCAB_PATH = "weights/vocab.pth"
```

`TOKENIZER_MODEL_PATH` must point to the `bangla_sentencepiece.model` generated by the same training run.

Then launch the CLI:

```powershell
cd model
python inference.py
```

Enter a Bangla sentence and one of the following target author names exactly:

- `bibhutibhushan bandopadhyay`
- `rabindranath tagore`
- `sarat chandra chattopadhyay`
- `satyajit ray`
- `sunil gangopadhay`

Enter `q` to exit. Sampling behaviour can be adjusted in `generate()` through `temperature`, `top_k_filter()`, and `repetition_penalty`.

## Data and Ethical Use

Only use source texts that you are legally permitted to process. The project is intended for research and educational experimentation in computational literary studies. Clearly label generated passages as machine-generated style-transfer output, avoid impersonation, and respect copyright, author estates, and cultural context.

## Project Materials

- [Project report](Project_Report.pdf)
- [Project slides](Project_Slides.pdf)

## Limitations

- The SentencePiece model improves local Bangla word dependencies, but its discriminator has not yet achieved the style-invariant chance-level accuracy reached by the word-level baseline.
- OCR quality and corpus size strongly affect the output.
- Generated output can still repeat tokens; decoding settings need further tuning for the subword vocabulary.
- The scripts are research-oriented and retain a few local configuration paths that must be adapted before use.
- The repository does not currently include a licence file; treat reuse as unlicensed until one is added by the project owner.
