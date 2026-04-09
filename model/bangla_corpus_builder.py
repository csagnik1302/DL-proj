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
from concurrent.futures import ProcessPoolExecutor, as_completed


# REGEX PRECOMPILED
RE_PAGE_NUM = re.compile(r'\n\s*\d+\s*\n')
RE_NUMBERS = re.compile(r'[\d০-৯]+')
RE_MULTISPACE = re.compile(r'\s+')
RE_NON_BANGLA = re.compile(r'[^\u0980-\u09FF\s।?!,;:\-\'\"()]')


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

def process_single_pdf(args):
    pdf_path, processed_files = args
    pdf_name = pdf_path.name

    if processed_files.get(pdf_name, False) is True:
        return None, pdf_name, "skipped"

    author_name = pdf_path.parent.name.lower()
    title_name = pdf_path.stem.lower()

    text = extract_text_pdfplumber(pdf_path)

    # retain your fallback logic EXACTLY
    if len(text) < 200 or not contains_bangla(text):
        text = extract_ocr_text(pdf_path)

    if len(text) < 300:
        return None, pdf_name, "too_short"

    sentences = split_sentences(text)

    doc = {
        "author": author_name,
        "title": title_name,
        "sentences": sentences[:1000]
    }

    return doc, pdf_name, "ok"


def load_processed_files(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed_files(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



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

    text = RE_PAGE_NUM.sub('\n', text)
    text = RE_NUMBERS.sub(' ', text)
    text = RE_MULTISPACE.sub(' ', text)
    text = RE_NON_BANGLA.sub(' ', text)
    text = RE_MULTISPACE.sub(' ', text)

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

    # Too many 1-letter words → OCR junk
    short_words = sum(1 for w in words if len(w) <= 2)
    if short_words / len(words) > 0.6:
        return False

    # Punctuation heavy
    punct = sum(1 for c in s if c in '.,:;!?-—()[]\'"')
    if punct / max(len(s),1) > 0.20:
        return False

    # Must contain Bengali vowel matras or vowels
    vowels = "ািীুূেৈোৌঅআইঈউঊএঐওঔ"
    if sum(c in vowels for c in s) < 3:
        return False

    # Repeated same char patterns (OCR artifacts)
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

def extract_text_pdfplumber(pdf_path):

    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

    except Exception as e:
        print("pdfplumber error:", e)

    return clean_text(text)




def extract_ocr_text(pdf_path):
    text = ""
    try:
        from pdf2image.pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        total_pages = info["Pages"]

        batch_size = 5  # 🔥 tune this (3–10 depending on RAM)

        for start in range(1, total_pages + 1, batch_size):
            end = min(start + batch_size - 1, total_pages)

            images = convert_from_path(
                pdf_path,
                dpi=300,
                first_page=start,
                last_page=end,
                poppler_path=POPPLER_PATH
            )

            for img in images:
                text += pytesseract.image_to_string(
                    img,
                    lang="ben",
                    config="--oem 3 --psm 6"
                )
                img.close()

            del images  # 🔥 force memory release

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

        self.processed_file_path = self.output_dir / "processed_files.json"
        self.processed_files = load_processed_files(self.processed_file_path)


    def process_folder(self, root_folder):
        pdf_files = list(Path(root_folder).rglob("*.pdf"))

        # Initialize unseen files as False
        for pdf in pdf_files:
            if pdf.name not in self.processed_files:
                self.processed_files[pdf.name] = False

        if not pdf_files:
            print("No PDF files found.")
            return

        with ProcessPoolExecutor(max_workers=2) as executor:

            futures = [
                executor.submit(process_single_pdf, (pdf, self.processed_files))
                for pdf in pdf_files
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                doc, pdf_name, status = future.result()

                if status == "ok":
                    print(f"PROCESSED: {pdf_name}")   # 👈 ADD THIS LINE
                    self.documents.append(doc)
                    self.processed_files[pdf_name] = True

                elif status == "skipped":
                    tqdm.write(f"Skipping already processed: {pdf_name}")

                elif status == "too_short":
                    tqdm.write(f"Skipped {pdf_name} (too short)")
                    self.processed_files[pdf_name] = True  # mark as done to avoid retry

        # ✅ ADD THIS BLOCK HERE (after executor finishes)
        save_processed_files(self.processed_file_path, self.processed_files)

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

        csv_path = self.output_dir / "sentences.csv"

        if csv_path.exists():
            old_df = pd.read_csv(csv_path)
            df_sents = pd.concat([old_df, df_sents], ignore_index=True)

        df_sents.to_csv(
            csv_path,
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

    builder = BanglaCorpusBuilder(output_dir=r"C:\project\DL project\Dataset_Creator\bangla_corpus")
    builder.process_folder(input_folder)
    builder.build()
