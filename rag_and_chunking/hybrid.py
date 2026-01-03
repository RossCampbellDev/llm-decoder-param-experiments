#!/usr/bin/env python3

from chunker import Chunk
from math import log
from typing import List

import re

"""
one of the primary challenges for RAG is "out of domain" work.
in-domain work would usually involve fine-tuned models working on a specific subject area (eg legal questions)
out-of-domain work is far more general, and is what we're experimenting with here

Hybrid search helps us produce better matches than simple dense vectors.  we combine our dense vector search
with sparse vector search.

we're going to work with BM25 (best-matching 25).  here is how it compares to the approach we have taken so far:
- Dense (embeddings) → meaning / paraphrase / intent
- BM25 → exact terms / sparse facts / existence queries

BM25 is a "bag of words" retrieval function
"""


def tokenize_chunk(chunk: Chunk) -> List[str]:
    # return chunk.txt.lower().split() # get words from a chunk
    return re.findall(r"[a-z0-9]+", chunk.txt.lower())


def tokenize_str(chunk: str) -> List[str]:
    # return chunk.lower().split()
    return re.findall(r"[a-z0-9]+", chunk.lower())

def tokenize_all(chunks: List[Chunk]) -> List[List[str]]:
    """
    we will most likely be passing the semantically-created chunks in here
    so that we can then produce BM25 stats
    """
    return [tokenize_chunk(c) for c in chunks]

 
def term_frequency(term: str, chunk: Chunk) -> int:
    """
    count how many times a search term occurs within a chunk
    """
    return tokenize_chunk(chunk).count(term)


def document_frequency(term: str, chunks: List[Chunk]) -> int:
    """
    really we should only be computing stuff once but this func does it on every call
    count how many chunks the search term occurs in in the entire document or collection
    """
    return sum(
        1 for c in chunks
        if term in tokenize_chunk(c)
    )


def inverse_document_frequency(term: str, chunks: List[Chunk]) -> float:
    """
    "how rare is the term across the doc/docs?"
    if a term is rare, the IDF will be high
    """
    df = document_frequency(term, chunks)
    return log(
        (len(chunks) - df + 0.5) / (df + 0.5) + 1
    )


