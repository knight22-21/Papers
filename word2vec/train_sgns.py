# train_sgns.py
import numpy as np
import collections
import random
import math
import pickle
from pathlib import Path
import sys
import time

def read_corpus(path):
    text = Path(path).read_text(encoding="utf8")
    words = text.split()
    return words

def build_vocab(words, vocab_size=None):
    counts = collections.Counter(words)
    if vocab_size:
        most_common = counts.most_common(vocab_size)
    else:
        most_common = counts.most_common()
    idx2word = [w for w, _ in most_common]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    freqs = np.array([c for _, c in most_common], dtype=np.float64)
    return word2idx, idx2word, freqs

def subsample(words, word2idx, freqs, t=1e-5):
    total = freqs.sum()
    prob_drop = {}
    for word, idx in word2idx.items():
        f = freqs[idx] / total
        prob_drop[word] = max(0.0, 1.0 - math.sqrt(t / f))
    out = []
    for w in words:
        if w in prob_drop and random.random() < prob_drop[w]:
            continue
        out.append(w)
    return out

def make_unigram_table(freqs, table_size=1_000_000, power=0.75):
    probs = freqs ** power
    probs = probs / probs.sum()
    table = np.zeros(table_size, dtype=np.int32)
    cumulative = np.cumsum(probs)
    j = 0
    for i in range(table_size):
        x = i / table_size
        while x > cumulative[j]:
            j += 1
        table[i] = j
    return table

def generate_training_data(words, word2idx, window_size):
    data = []
    N = len(words)
    for i, w in enumerate(words):
        if w not in word2idx:
            continue
        wi = word2idx[w]
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue
            wj = words[j]
            if wj not in word2idx:
                continue
            data.append((wi, word2idx[wj]))
    return data

# numerically stable sigmoid
def sigmoid(x):
    x = np.clip(x, -6, 6)
    return 1.0 / (1.0 + np.exp(-x))

def train_skipgram_ns(data, vocab_size, embed_dim=100, lr=0.025,
                      negative_samples=5, epochs=1, table=None, batch_size=1):
    W = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    data_len = len(data)

    print(f"Total training pairs: {data_len:,}")
    print("Starting training...\n")

    for epoch in range(epochs):
        random.shuffle(data)
        start_time = time.time()
        for i in range(0, data_len, batch_size):
            batch = data[i:i+batch_size]
            for center, context in batch:
                v_c = W[center]
                # positive example
                score = np.dot(W_out[context], v_c)
                g = (1.0 - sigmoid(score)) * lr
                W_out[context] += g * v_c
                W[center] += g * W_out[context]
                # negative samples
                for _ in range(negative_samples):
                    if table is not None:
                        neg = int(np.random.choice(table))
                    else:
                        neg = random.randrange(vocab_size)
                    if neg == context:
                        continue
                    score = np.dot(W_out[neg], v_c)
                    g = (0.0 - sigmoid(score)) * lr
                    W_out[neg] += g * v_c
                    W[center] += g * W_out[neg]

            # --- progress print every 100k steps ---
            if (i // batch_size) % 100000 == 0 and i > 0:
                pct = (i / data_len) * 100
                elapsed = time.time() - start_time
                sys.stdout.write(f"\rEpoch {epoch+1}/{epochs}: {pct:.2f}% done, elapsed {elapsed:.1f}s")
                sys.stdout.flush()

        # simple linear decay of learning rate
        lr = lr * (1.0 - epoch / max(1, epochs))
        if lr < 0.00001:
            lr = 0.00001
        elapsed_epoch = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} completed in {elapsed_epoch:.1f}s, lr={lr:.6f}")

    print("\nTraining finished.")
    return W, W_out

def save_embeddings(path, W, idx2word):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((W, idx2word), f)
    print(f"Embeddings saved to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="path to corpus file")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print("Reading corpus")
    words = read_corpus(args.corpus)
    print("Building vocab")
    word2idx, idx2word, freqs = build_vocab(words, vocab_size=args.vocab_size)
    print("Subsampling frequent words")
    words_sub = subsample(words, word2idx, freqs)
    print("Make unigram table")
    table = make_unigram_table(freqs[:len(idx2word)])
    print("Generate training data")
    data = generate_training_data(words_sub, word2idx, args.window)
    print("Training skip gram with negative sampling")
    W, W_out = train_skipgram_ns(data, len(idx2word), embed_dim=args.embed_dim,
                                 lr=0.025, negative_samples=args.neg, epochs=args.epochs, table=table)
    print("Saving embeddings")
    save_embeddings("embeddings/embeddings.pkl", W, idx2word)
    print("Done")
