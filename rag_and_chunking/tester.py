#!/usr/bin/env python3

from embedder import best_match
from chunker import Chunk
from semantic_chunker import semantic_chunks
from hybrid import tokenize_chunk, inverse_document_frequency, tokenize_str
from typing import List, Tuple
import sys


def load_test_data(path: str = "test_doc.md") -> str:
    with open(path, "r") as f:
        return f.read().strip()


def build_chunks(test_data: str) -> List[Chunk]:
    # chunks = structure_aware(test_data)
    # chunks = fixed_chunks(test_data, 128, 16)
    return semantic_chunks(test_data)


def compute_dense_scores(query: str, chunks: List[Chunk]) -> List[Tuple[int, float]]:
    return sorted(
        best_match(query, chunks),
        key=lambda x: x[1],
        reverse=True
    )

# print(f"Query:\n\t{query}")
# 
# print("\nsemantic chunk dense vector top 5")
# for i, score in dense_scores[:5]:
#     print("\t", score, chunks[i].txt)


def bm25_score(tokenized_query: List[str], chunk: Chunk, chunks: List[Chunk], avgdl: float) -> float:
    tokenized_chunk = tokenize_chunk(chunk)

    # these params control how quickly TF stops mattering, and how aggressively
    # long chunks are penalized
    k1 = 1.2 # saturation parameter.  term frequency saturation
    b = 0.75 # length normalisation

    score = 0.0

    for term in set(tokenized_query): # use a set to prevent repeating terms being over-weighted!
       tf = tokenized_chunk.count(term)
       if tf == 0:
           continue
       idf = inverse_document_frequency(term, chunks)
       dl = lambda x: len(tokenize_chunk(x))
       denominator = tf + k1 * (1 - b + b * (dl(chunk) / avgdl))
       score += idf * (tf * (k1 + 1)) / denominator

    return score


def normalise(scores: List[float]) -> List[float]:
    m = max(scores)
    return [s / m if m > 0 else 0.0 for s in scores]


def get_hybrid(query: str, path: str = "test_doc.md") -> List[Tuple[int, float]]:
    test_data = load_test_data(path)
    chunks = build_chunks(test_data)

    dense_scores = compute_dense_scores(query, chunks)

    query_tokens = tokenize_str(query)
    dl = lambda x: len(tokenize_chunk(x))
    avgdl = sum(dl(c) for c in chunks) / len(chunks)

    bm25_scores = []
    for i, c in enumerate(chunks):
        s = bm25_score(query_tokens, c, chunks, avgdl)
        bm25_scores.append((i, s))

    bm25_scores.sort(key=lambda x: x[1], reverse=True)

    dense_scores_by_chunk = [0.0] * len(chunks)
    for i, s in dense_scores:
        dense_scores_by_chunk[i] = s

    sparse_scores_by_chunk = [0.0] * len(chunks)
    for i, s in bm25_scores:
        sparse_scores_by_chunk[i] = s

    dense_scores_by_chunk = normalise(dense_scores_by_chunk)
    sparse_scores_by_chunk = normalise(sparse_scores_by_chunk)

    alpha = 0.5
    hybrid_scores = [
        alpha * d + (1 - alpha) * b
        for d, b in zip(dense_scores_by_chunk, sparse_scores_by_chunk)
    ]

    return sorted(
        enumerate(hybrid_scores),
        key=lambda x: x[1],
        reverse=True
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("supply a prompt please")
    get_hybrid(sys.argv[1])
