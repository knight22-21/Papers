import os
import math
import random
import argparse
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# ------------------------
# Utilities and preprocessing
# ------------------------

def read_corpus(path, lowercase=True):
    """Yield tokens from a corpus file."""
    with open(path, encoding='utf8') as f:
        for line in f:
            if lowercase:
                line = line.lower()
            tokens = line.strip().split()
            if tokens:
                for t in tokens:
                    yield t


def build_vocab(tokens, min_count=5, max_vocab=None):
    ctr = Counter(tokens)
    items = [(w, c) for w, c in ctr.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    if max_vocab is not None:
        items = items[:max_vocab]
    idx2word = [w for w, c in items]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    counts = [c for w, c in items]
    return word2idx, idx2word, counts


class CoOccurrenceBuilder:
    """Construct sparse co occurrence counts using symmetric window and 1/d weighting."""
    def __init__(self, word2idx, window_size=5):
        self.word2idx = word2idx
        self.window_size = window_size
        self.cooccurrence = defaultdict(float)
        self.total = 0

    def add_sentence(self, sentence_tokens):
        indices = [self.word2idx[t] for t in sentence_tokens if t in self.word2idx]
        L = len(indices)
        for center_pos, center_id in enumerate(indices):
            start = max(0, center_pos - self.window_size)
            end = min(L, center_pos + self.window_size + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                context_id = indices[context_pos]
                distance = abs(center_pos - context_pos)
                if distance == 0:
                    continue
                weight = 1.0 / distance
                self.cooccurrence[(center_id, context_id)] += weight
                self.total += weight

    def items(self):
        return list(self.cooccurrence.items())


# ------------------------
# PyTorch dataset for non zero co occurrence pairs
# ------------------------
class CooccurDataset(Dataset):
    def __init__(self, cooccurrence_items, xmax=100, alpha=0.75):
        self.pairs = [pair for pair, count in cooccurrence_items]
        self.counts = [count for pair, count in cooccurrence_items]
        self.xmax = xmax
        self.alpha = alpha

    def __len__(self):
        return len(self.pairs)

    def weight(self, x):
        return (x / self.xmax) ** self.alpha if x < self.xmax else 1.0

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        x = self.counts[idx]
        w = self.weight(x)
        return i, j, x, w


# ------------------------
# GloVe model in PyTorch
# ------------------------
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.wi = nn.Embedding(vocab_size, emb_dim)
        self.wj = nn.Embedding(vocab_size, emb_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.5 / self.emb_dim
        nn.init.uniform_(self.wi.weight, -init_range, init_range)
        nn.init.uniform_(self.wj.weight, -init_range, init_range)
        nn.init.constant_(self.bi.weight, 0.0)
        nn.init.constant_(self.bj.weight, 0.0)

    def forward(self, i_idx, j_idx):
        vi = self.wi(i_idx)
        vj = self.wj(j_idx)
        bi = self.bi(i_idx).squeeze(1)
        bj = self.bj(j_idx).squeeze(1)
        dot = torch.sum(vi * vj, dim=1)
        return dot, bi, bj


# ------------------------
# Helper functions
# ------------------------
def save_embeddings(model, idx2word, output_path):
    wi = model.wi.weight.data.cpu()
    wj = model.wj.weight.data.cpu()
    embeddings = wi + wj

    os.makedirs(output_path, exist_ok=True)
    emb_file = os.path.join(output_path, 'glove_embeddings.txt')
    print(f'Saving embeddings to {emb_file}')
    with open(emb_file, 'w', encoding='utf8') as f:
        for i, w in enumerate(idx2word):
            vec = embeddings[i].numpy()
            vec_str = ' '.join(map(str, vec))
            f.write(f'{w} {vec_str}\n')
    print('Done saving embeddings.')


def check_nearest_neighbors(model, idx2word, top_k=5):
    wi = model.wi.weight.data.cpu().numpy()
    idx = random.randint(0, len(idx2word) - 1)
    word = idx2word[idx]
    sims = cosine_similarity([wi[idx]], wi)[0]
    top_idx = sims.argsort()[-top_k-1:-1][::-1]
    print(f'\nNearest neighbors for "{word}":')
    for i in top_idx:
        print(f'{idx2word[i]} ({sims[i]:.4f})')


def evaluate_analogy(model, word2idx, idx2word):
    wi = model.wi.weight.data.cpu().numpy()
    def get_vec(w):
        return wi[word2idx[w]] if w in word2idx else None

    example = ('king', 'man', 'woman')
    a, b, c = map(get_vec, example)
    
    # Check if any vector is None
    if any(vec is None for vec in (a, b, c)):
        print("Analogy words not in vocab, skipping.")
        return
    
    target = a - b + c
    sims = cosine_similarity([target], wi)[0]
    best = sims.argsort()[-6:][::-1]
    print("\nAnalogy: king - man + woman = ?")
    for i in best:
        print(f"{idx2word[i]} ({sims[i]:.4f})")

# ------------------------
# Training Loop
# ------------------------
def train_glove(corpus_path, output_path, emb_dim=100, window_size=5, min_count=5,
                xmax=100, alpha=0.75, epochs=25, batch_size=1024, lr=0.05, max_vocab=50000):

    print('Reading corpus and building vocabulary...')
    tokens = list(read_corpus(corpus_path))
    word2idx, idx2word, counts = build_vocab(tokens, min_count=min_count, max_vocab=max_vocab)
    vocab_size = len(idx2word)
    print(f'Vocab size: {vocab_size}')

    print('Building co-occurrence matrix...')
    builder = CoOccurrenceBuilder(word2idx, window_size=window_size)
    with open(corpus_path, encoding='utf8') as f:
        for line in f:
            if line.strip():
                builder.add_sentence(line.lower().strip().split())

    co_items = builder.items()
    print(f'Non-zero co-occurrence pairs: {len(co_items)}')

    dataset = CooccurDataset(co_items, xmax=xmax, alpha=alpha)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GloVeModel(vocab_size, emb_dim).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

    log_dir = "runs/glove_training"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n🔍 TensorBoard logs being written to: {log_dir}\n")

    print('Starting training...\n')
    for epoch in range(epochs):
        total_loss = 0.0
        start_time = time.time()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for i, (i_idx, j_idx, x_ij, weight) in pbar:
            i_idx, j_idx = i_idx.to(device), j_idx.to(device)
            x_ij, weight = x_ij.to(device), weight.to(device)

            optimizer.zero_grad()
            dot, bi, bj = model(i_idx, j_idx)
            log_x = torch.log(x_ij)
            diff = dot + bi + bj - log_x
            loss = torch.sum(weight * diff ** 2) / i_idx.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * i_idx.size(0)

            # Log every 1000 steps for live TensorBoard updates
            if i % 1000 == 0:
                writer.add_scalar('Loss/batch', loss.item(), epoch * len(dataloader) + i)

            # ETA estimation
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (i + 1)
            remaining_batches = len(dataloader) - (i + 1)
            eta = remaining_batches * avg_time_per_batch
            pbar.set_postfix(loss=f"{loss.item():.4f}", ETA=f"{eta/60:.1f} min")

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.6f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch)

        if (epoch + 1) % 5 == 0:
            check_nearest_neighbors(model, idx2word)

    writer.close()
    save_embeddings(model, idx2word, output_path)
    evaluate_analogy(model, word2idx, idx2word)
    print("\nTraining complete.")


# ------------------------
# CLI
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='Path to corpus text file')
    parser.add_argument('--out', type=str, default='output', help='Output directory')
    parser.add_argument('--dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--window', type=int, default=5, help='Context window size')
    parser.add_argument('--min_count', type=int, default=5, help='Min frequency to keep token')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--max_vocab', type=int, default=50000)
    args = parser.parse_args()

    train_glove(args.corpus, args.out, emb_dim=args.dim, window_size=args.window,
                min_count=args.min_count, epochs=args.epochs, batch_size=args.batch,
                lr=args.lr, max_vocab=args.max_vocab)
