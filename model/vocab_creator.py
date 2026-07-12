from pathlib import Path
from tempfile import NamedTemporaryFile

import sentencepiece as spm


TOKENIZER_MODEL_PATH = Path(__file__).with_name("bangla_sentencepiece.model")
VOCAB_SIZE = 8000


def _train_sentencepiece(data, model_path=TOKENIZER_MODEL_PATH):
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as corpus_file:
        for sentence in data:
            text = str(sentence).strip()
            if text:
                corpus_file.write(f"{text}\n")
        corpus_path = Path(corpus_file.name)

    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_path.with_suffix("")),
            vocab_size=VOCAB_SIZE,
            model_type="unigram",
            character_coverage=1.0,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece="<unk>",
            bos_piece="<sos>",
            eos_piece="<eos>",
            pad_piece="<pad>",
            hard_vocab_limit=False,
        )
    finally:
        corpus_path.unlink(missing_ok=True)


def get_sentencepiece_processor(data=None, model_path=TOKENIZER_MODEL_PATH):
    if not model_path.exists():
        if data is None:
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path}. Train the model before inference."
            )
        _train_sentencepiece(data, model_path)

    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    return processor


# SentencePiece tokenizer (active)
def tokenize(data):
    processor = get_sentencepiece_processor(data)
    return [processor.encode(str(sentence), out_type=int) + [processor.eos_id()] for sentence in data]


def vocab(data):
    processor = get_sentencepiece_processor(data)
    return {processor.id_to_piece(index): index for index in range(processor.get_piece_size())}


def vocab_creator(data):
    return vocab(data)


# Word-level tokenizer (legacy fallback; keep commented for future experiments)
# def tokenize(data):
#     token_set = []
#     for sentence in data:
#         tokens = sentence.split()
#         tokens.append("<eos>")
#         token_set.append(tokens)
#     return token_set
#
#
# def vocab(data):
#     vocabulary = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<sos>": 3}
#     count = Counter(token for sentence in data for token in sentence)
#     for token, frequency in count.items():
#         if frequency > 5 and token not in vocabulary:
#             vocabulary[token] = len(vocabulary)
#     return vocabulary


if __name__ == "__main__":
    import pandas as pd

    dataset = pd.read_csv(r"C:\project\DL project\Dataset_Creator\bangla_corpus\sentences.csv")
    print(tokenize(dataset["text"])[:5])
