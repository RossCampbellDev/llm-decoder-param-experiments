#!/usr/bin/env python3

from embedder import embed
from chunker import Chunk
from typing import List
import numpy as np
import re


# thanks gpt
def sentence_split(text: str) -> List[str]:
    sentences: List[str] = []
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def semantic_chunks(raw_text: str, threshold=0.75) -> List[Chunk]:
    chunks: List[Chunk] = []

    # Split text into sentences
    sentences = sentence_split(raw_text)
    # Embed each sentence
    embedded_sentences = [embed(s) for s in sentences]
    
    # Accumulate sentences into a chunk while
    # cosine(avg_chunk_embedding, next_sentence_embedding) > threshold
    # Cut when similarity drops
    current_sentences = [sentences[0]]
    current_vector = embedded_sentences[0]

    # iterate over our sentences.  evaluate the similarity of each new sentence to the accumulated chunk so far
    # if the similarity meets the threshold, add it to the current chunk's vector and mean
    # else, start a new chunk
    # note: start with `sentences[1:], embedded_sentences[1:]` because we already start with the 0th values
    for sentence, vector in zip(sentences, embedded_sentences):
        similarity = current_vector @ vector
        if similarity > threshold:
            current_sentences.append(sentence)

            # adapt the current vector to include the 'meaning' of the newest embedded sentence
            current_vector = np.mean(
                [current_vector, vector],
                axis=0
            )
            # normalize the result to maintain magnitude = 1
            current_vector /= np.linalg.norm(current_vector)
        else:
            chunks.append(
                Chunk(txt='. '.join(current_sentences), meta={})
            )
            current_sentences = [sentence]
            current_vector = vector

    chunks.append(
        Chunk(txt='. '.join(current_sentences), meta={})
    )

    return chunks
