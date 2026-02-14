import pandas as pd
import re
import unicodedata
from collections import defaultdict

# ======================
# Bangla Quality Metrics
# ======================

DEPENDENT_SIGNS = set("ািীুূৃেৈোৌ্")

def count_isolated_signs(text):
    words = text.split()
    return sum(1 for w in words if w and w[0] in DEPENDENT_SIGNS)

def single_char_ratio(text):
    tokens = text.split()
    if not tokens:
        return 0
    return sum(1 for w in tokens if len(w) == 1) / len(tokens)

def non_bangla_ratio(text):
    total = len(text)
    if total == 0:
        return 1
    non_bangla = sum(
        1 for c in text
        if not (0x0980 <= ord(c) <= 0x09FF or c.isspace())
    )
    return non_bangla / total

def compute_noise_score(text):
    iso = count_isolated_signs(text)
    single_ratio = single_char_ratio(text)
    non_ratio = non_bangla_ratio(text)

    score = 0
    score += iso * 0.01
    score += single_ratio * 2
    score += non_ratio * 3

    return score


# ======================
# Corpus Validator
# ======================

def validate_corpus(csv_path):

    df = pd.read_csv(csv_path)

    print("\n==============================")
    print("CORPUS VALIDATION REPORT")
    print("==============================\n")

    print("Total sentences:", len(df))
    print("Unique authors:", df["author"].nunique())

    # Word statistics
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    total_words = df["word_count"].sum()

    print("Total words in corpus:", total_words)
    print("Average words per sentence:", round(df["word_count"].mean(), 2))

    # ======================
    # OCR Noise Metrics
    # ======================

    print("\nComputing OCR quality metrics...\n")

    df["noise_score"] = df["text"].apply(lambda x: compute_noise_score(str(x)))

    print("Average noise score (corpus):", round(df["noise_score"].mean(), 3))
    print("High-noise sentences:", len(df[df["noise_score"] > 2]))

    # ======================
    # Author-Level Statistics
    # ======================

    print("\n==============================")
    print("AUTHOR-LEVEL STATISTICS")
    print("==============================\n")

    author_groups = df.groupby("author")

    for author, group in author_groups:

        author_sentences = len(group)
        author_words = group["word_count"].sum()
        avg_words = group["word_count"].mean()
        avg_noise = group["noise_score"].mean()
        vocab_size = len(set(" ".join(group["text"]).split()))

        print(f"Author: {author}")
        print(f"  Sentences: {author_sentences}")
        print(f"  Total words: {author_words}")
        print(f"  Avg words/sentence: {round(avg_words, 2)}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Avg OCR noise: {round(avg_noise, 3)}")
        print("-" * 40)

    print("\n==============================")
    print("Validation complete.")
    print("==============================\n")


# ======================
# Run
# ======================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python corpus_validator.py <sentences.csv>")
    else:
        validate_corpus(sys.argv[1])
