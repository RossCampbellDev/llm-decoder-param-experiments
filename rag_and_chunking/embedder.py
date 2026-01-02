#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
from typing import cast, List, Union, Tuple
from chunker import Chunk
import numpy as np

# use a sentence embedding model, not our main generation model
# something like BERT
# small, fast, bidirectional.  takes input and returns one vector

# for more info see ttps://huggingface.co/sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(sentences: Union[str, Chunk, List[str], List[Chunk]]) -> np.ndarray:
    """
    takes chunks or strings and runs them through the sentence embedding model's encoder
    returns one vector containing all the chunk vectors
    """
    if isinstance(sentences, list):
        if isinstance(sentences[0], Chunk):
            chunks = cast(list[Chunk], sentences)
            sentences = [c.txt for c in chunks]
        sentences = cast(list[str], sentences)
    else:
        if isinstance(sentences, Chunk):
            sentences = sentences.txt
    result = model.encode(sentences, normalize_embeddings=True)
    return result


def best_match(query: str, sentences: Union[List[str], List[Chunk]]) -> List[Tuple[int, float]]:
    """
    takes a query string and a list of chunks or strings, embeds them into vectors,
    then uses the model's similarity function to pick the highest probability answer
    """
    if isinstance(sentences, list):
        if isinstance(sentences[0], Chunk):
            chunks = cast(list[Chunk], sentences)
            sentences = [c.txt for c in chunks]
        sentences = cast(list[str], sentences)

    result = my_similarity(query, sentences)

    return list(enumerate(result))

def my_similarity(query: str, sentences: List[str]) -> np.ndarray:
    """
    very basic cosine similarity check.  the equation is usually:
        dot(a, b) = ||a|| * ||b||
    however, we already enforced normalisation in our `model.encode`
    which essentially simplifies the procedure to dotproduct
    """
    q = embed(query)
    # shape is not necessarily always `d` for query, so check it:
    if q.ndim == 2:
        q = q[0]
    x = embed(sentences)
    scores = x @ q

    return scores
