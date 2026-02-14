import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

import pandas as pd
import unicodedata


# =========================
# SET TESSERACT PATH               
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # <---- add your own path of tesseract installation

# =========================
# SET POPPLER PATH (ADD THIS)
# =========================
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"


# =========================
# Bangla Text Utilities
# =========================

def contains_bangla(text):
    return any(0x0980 <= ord(c) <= 0x09FF for c in text)

def linguistic_score(word: str) -> float:
    """Higher score = more likely real Bangla language"""

    if not word:
        return 0

    # Bangla chars
    bangla = sum(0x0980 <= ord(c) <= 0x09FF for c in word) / len(word)

    # vowel matras presence (very important in Bangla)
    vowels = "ািীুূেৈোৌঅআইঈউঊএঐওঔ"
    vowel_ratio = sum(c in vowels for c in word) / len(word)

    # longer words more likely real
    length_bonus = min(len(word) / 6, 1)

    return 0.5 * bangla + 0.4 * vowel_ratio + 0.1 * length_bonus

def cut_to_linguistic_region(s: str) -> str:

    words = s.split()
    if len(words) < 6:
        return s

    window = 4

    for i in range(len(words) - window):
        segment = words[i:i+window]
        score = sum(linguistic_score(w) for w in segment) / window

        # threshold experimentally good for Bengali OCR
        if score > 0.45:
            return " ".join(words[i:])

    return s


def clean_text(text):

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters
    text = text.replace('\u200c', '')
    text = text.replace('\u200d', '')

    # Remove page-number-only lines
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove isolated numbers surrounded by spaces
    text = re.sub(r'[\d০-৯]+', ' ', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Keep Bangla + punctuation only
    text = re.sub(r'[^\u0980-\u09FF\s।?!,;:\-\'\"()]', ' ', text)

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def trim_ocr_edges(s: str) -> str:

    tokens = s.split()

    def good_token(w):
        # at least 2 Bangla letters
        bangla_letters = sum(0x0980 <= ord(c) <= 0x09FF for c in w)
        return bangla_letters >= 2 and not re.fullmatch(r"[\"'():;,\-]+", w)

    # trim start
    while tokens and not good_token(tokens[0]):
        tokens.pop(0)

    # trim end
    while tokens and not good_token(tokens[-1]):
        tokens.pop()

    return " ".join(tokens)



def is_valid_bangla_sentence(s: str) -> bool:

    words = s.split()
    if len(words) < 5:
        return False

    # 1️⃣ Too many 1-letter words → OCR junk
    short_words = sum(1 for w in words if len(w) <= 2)
    if short_words / len(words) > 0.6:
        return False

    # 2️⃣ Punctuation heavy
    punct = sum(1 for c in s if c in '.,:;!?-—()[]\'"')
    if punct / max(len(s),1) > 0.20:
        return False

    # 3️⃣ Must contain Bengali vowel matras or vowels
    vowels = "ািীুূেৈোৌঅআইঈউঊএঐওঔ"
    if sum(c in vowels for c in s) < 3:
        return False

    # 4️⃣ Repeated same char patterns (OCR artifacts)
    if re.search(r'(.)\1\1\1', s):
        return False

    return True



def split_sentences(text):
    sentences = re.split(r'[।?!]', text)

    cleaned = []

    for s in sentences:
        s = s.strip()

        # Skip very short lines
        if len(s.split()) < 5:
            continue

        # Skip sentences that are mostly digits
        if re.fullmatch(r'\d+', s):
            continue

        # Skip lines that are mostly numeric or noise
        digit_ratio = sum(c.isdigit() for c in s) / max(len(s), 1)
        if digit_ratio > 0.3:
            continue

        s = trim_ocr_edges(s)
        s = cut_to_linguistic_region(s)

        if is_valid_bangla_sentence(s):
            cleaned.append(s)

    return cleaned



# =========================
# OCR Extraction (Memory Safe)
# =========================

def extract_ocr_text(pdf_path):
    text = ""
    try:
        from pdf2image.pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        total_pages = info["Pages"]

        for page_number in tqdm(
            range(1, total_pages + 1),
            desc=f"OCR pages: {Path(pdf_path).name}",
            leave=False
        ):
            images = convert_from_path(
                pdf_path,
                dpi=300,
                first_page=page_number,
                last_page=page_number,
                poppler_path=POPPLER_PATH
            )

            img = images[0]

            text += pytesseract.image_to_string(
                img,
                lang="ben",
                config="--oem 3 --psm 6"
            )

            img.close()
            del img
            del images

    except Exception as e:
        tqdm.write(f"OCR error in {pdf_path}: {e}")

    return clean_text(text)


# =========================
# Corpus Builder
# =========================

class BanglaCorpusBuilder:

    def __init__(self, output_dir="bangla_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.documents = []

    def process_folder(self, root_folder):
        pdf_files = list(Path(root_folder).rglob("*.pdf"))

        if not pdf_files:
            print("No PDF files found.")
            return

        for pdf in tqdm(pdf_files, desc="Processing PDFs"):
            author_name = pdf.parent.name.lower()
            title_name = pdf.stem.lower()

            tqdm.write(f"OCR used for: {pdf.name}")
            text = extract_ocr_text(pdf)

            tqdm.write(f"{pdf.name} -> extracted length: {len(text)}")

            if len(text) < 300:
                tqdm.write(f"Skipped {pdf.name} (too short)")
                continue

            sentences = split_sentences(text)

            doc = {
                "author": author_name,
                "title": title_name,
                "sentences": sentences[:500]
            }

            self.documents.append(doc)

        print(f"\nTotal documents processed: {len(self.documents)}")

    def build(self):

        if not self.documents:
            print("No valid documents found.")
            return

        # Sentence-level CSV only
        sentence_rows = []

        for doc in self.documents:
            for sentence in doc["sentences"]:
                sentence_rows.append({
                    "author": doc["author"],
                    "text": sentence
                })

        df_sents = pd.DataFrame(sentence_rows)

        df_sents.to_csv(
            self.output_dir / "sentences.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nCorpus successfully built")
        print(f"Total sentences: {len(df_sents)}")
        print(f"Unique authors: {df_sents['author'].nunique()}")
        print(f"Output directory: {self.output_dir}")


# =========================
# Run Script
# =========================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python bangla_corpus_builder.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]

    builder = BanglaCorpusBuilder()
    builder.process_folder(input_folder)
    builder.build()
