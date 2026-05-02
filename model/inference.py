import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit these two lines
# ─────────────────────────────────────────────
WEIGHTS_PATH = r"weights.pth"
VOCAB_PATH   = r"vocab.pth"


# ─────────────────────────────────────────────
# Author name  →  class index  (must match training)
# ─────────────────────────────────────────────
AUTHOR_MAP = {
    "bibhutibhushan bandopadhyay": 0,
    "rabindranath tagore":         1,
    "sarat chandra chattopadhyay": 2,
    "satyajit ray":                3,
    "sunil gangopadhay":           4,
}


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Load vocab
# ─────────────────────────────────────────────
vocab: dict = torch.load(VOCAB_PATH, map_location="cpu")
# Reverse vocab: index → word  (for decoding generated token IDs back to text)
idx_to_word = {idx: word for word, idx in vocab.items()}


# ─────────────────────────────────────────────
# Rebuild model architecture
# IMPORTANT: these hyperparameters must be identical to Adversary_Passes.py
# ─────────────────────────────────────────────
embed = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=128,
    padding_idx=vocab["<pad>"]
).to(device)

emb_author = nn.Embedding(5, embedding_dim=128).to(device)

# Encoder (bidirectional)
encoder_gru = nn.GRU(
    input_size=128,
    hidden_size=130,
    batch_first=True,
    bidirectional=True
).to(device)

# Decoder — unidirectional GRU
# Input: word_embed(128) + encoder_context(260) + author_embed(128) = 516
# Output per timestep: 130 (single forward direction)
decoder_gru = nn.GRU(
    input_size=128 + 260 + 128,
    hidden_size=130,
    batch_first=True,
    bidirectional=False
).to(device)

# Projection: 130 → vocab_size
decoder_linear = nn.Linear(130, len(vocab), bias=True).to(device)


# ─────────────────────────────────────────────
# Load trained weights
# ─────────────────────────────────────────────
weights = torch.load(WEIGHTS_PATH, map_location=device)

embed.load_state_dict(weights["embed"])
encoder_gru.load_state_dict(weights["encoder_gru"])
decoder_gru.load_state_dict(weights["decoder_gru"])
decoder_linear.load_state_dict(weights["decoder_linear"])
emb_author.load_state_dict(weights["emb_author"])

print("Weights loaded successfully.\n")

# Switch all modules to eval mode:
# - disables dropout (if any)
# - makes BatchNorm use running stats (if any)
# - most importantly: signals no gradient tracking needed
embed.eval()
encoder_gru.eval()
decoder_gru.eval()
decoder_linear.eval()
emb_author.eval()


# ─────────────────────────────────────────────
# Helper: tokenize + encode a single sentence
# ─────────────────────────────────────────────
def encode_sentence(sentence: str) -> torch.Tensor:
    """
    Splits sentence into words, prepends <sos>, appends <eos>, maps each word to its vocab
    index (<unk> for out-of-vocabulary words), returns shape (1, seq_len).
    """
    words = ["<sos>"] + sentence.strip().split() + ["<eos>"]
    indices = [vocab.get(w, vocab["<unk>"]) for w in words]
    # shape: (1, seq_len)  — batch dimension added so it matches model expectations
    return torch.tensor([indices], dtype=torch.long).to(device)


# ─────────────────────────────────────────────
# Helper: run encoder, return hidden_both
# ─────────────────────────────────────────────

def top_k_filter(logits, k=20):
    # Zero out all logits except the top-k
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(1)
    logits = torch.where(logits < min_val, 
                         torch.full_like(logits, float('-inf')), 
                         logits)
    return logits



def encode(token_ids: torch.Tensor) -> torch.Tensor:
    """
    token_ids: (1, seq_len)
    Returns hidden_both: (1, 260)  — forward_h concat backward_h
    """
    embedded = embed(token_ids)             # (1, seq_len, 128)
    _, hidden = encoder_gru(embedded)       # hidden: (2, 1, 130)  [fwd, bwd]
    hidden_both = torch.cat(
        (hidden[0], hidden[1]), dim=1
    )                                       # (1, 260)
    return hidden_both


# ─────────────────────────────────────────────
# Helper: autoregressive generation
# ─────────────────────────────────────────────
def generate(hidden_both: torch.Tensor,
             author_idx: int,
             max_len: int = 30,
             temperature: float = 0.7,        # add this
             repetition_penalty: float = 1.5  # add this
             ) -> list[str]:

    i_auth = torch.tensor([author_idx], dtype=torch.long).to(device)
    author_vec = emb_author(i_auth)
    sos = vocab["<sos>"]
    input_token = torch.tensor([[sos]], dtype=torch.long).to(device)

    hidden = None
    generated = []
    generated_indices = []   # track token ids for repetition penalty


    context = hidden_both.unsqueeze(1)


    with torch.no_grad():
        for _ in range(max_len):

            word_emb = embed(input_token)
            style    = author_vec.unsqueeze(1)

            dec_input = torch.cat((word_emb, context, style), dim=2)
            output, hidden = decoder_gru(dec_input, hidden)
            # output: (1, 1, 130) — single forward direction, no slicing needed

            logits = decoder_linear(output.squeeze(1))   # (1, vocab_size)

            # ── Block special tokens ──────────────────────────────────
            for special in ["<pad>", "<unk>", "<sos>"]:
                logits[0, vocab[special]] = float('-inf')

            # ── Repetition penalty ───────────────────────────────────
            # Divide logits of already-generated tokens by penalty factor
            # making them less likely to be picked again
            for prev_idx in set(generated_indices):
                logits[0, prev_idx] /= repetition_penalty

            # ── Temperature sampling ─────────────────────────────────
            # Dividing by temperature < 1 sharpens the distribution (more confident)
            # Dividing by temperature > 1 flattens it (more random)
            # 0.7-0.9 is a good range — creative but not chaotic
            logits = logits / temperature
            logits = top_k_filter(logits, k=20)   # add this line
            probs  = torch.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == vocab["<eos>"]:
                break

            word = idx_to_word.get(next_idx, "<unk>")
            generated.append(word)
            generated_indices.append(next_idx)

            input_token = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return generated


# ─────────────────────────────────────────────
# Main interactive loop
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bangla Style Transfer — Inference")
    print("=" * 55)
    print("Available target authors:")
    for name, idx in AUTHOR_MAP.items():
        print(f"  [{idx}]  {name}")
    print()

    while True:
        # ── Input sentence ───────────────────────────────
        sentence = input("Enter a Bangla sentence (or 'q' to quit):\n> ").strip()
        if sentence.lower() == "q":
            break

        # ── Target author ────────────────────────────────
        author_input = input("Target author name: ").strip().lower()
        if author_input not in AUTHOR_MAP:
            print(f"Unknown author. Choose from: {list(AUTHOR_MAP.keys())}\n")
            continue

        author_idx = AUTHOR_MAP[author_input]

        # ── Encode input ─────────────────────────────────
        token_ids   = encode_sentence(sentence)
        hidden_both = encode(token_ids)

        # ── Generate ─────────────────────────────────────
        words = generate(hidden_both, author_idx, temperature=0.7, repetition_penalty=1.5)
        output = " ".join(words) if words else "(no output — model may need more training)"

        print(f"\nGenerated ({author_input}):\n  {output}\n")
        print("-" * 55)


if __name__ == "__main__":
    main()