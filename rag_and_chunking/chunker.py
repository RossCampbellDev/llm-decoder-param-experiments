#!/usr/bin/env python3

import re

from typing import List

class Chunk:
    txt: str
    meta: dict

    def __init__(self, txt, meta):
        self.txt = txt
        self.meta = meta

# fixed size chunks
def fixed_chunks(txt: str, chunk_size: int, overlap: int=0) -> List[Chunk]:
    chunks: List[Chunk] = []

    i = 0
    while (i + chunk_size) <= len(txt):
        chunks.append(
            Chunk(
                txt=txt[i:i+chunk_size].replace("\n", ""),
                meta={}
            )
        )
        i = i + (chunk_size - overlap)

    return chunks


def structure_aware(text: str) -> List[Chunk]:
    chunks: List[Chunk] = []

    md_re = [
            re.compile(r"<h[0-9][\sA-z]*>.*"), # HTML headers
            re.compile(r".*:.*"), # paragraph breaks (this is very aggressive and likely flawed)
            re.compile(r"#{1,4}\s.*"), # markdown headers
    ]
    current_chunk_lines: List[str] = []

    for line in text.splitlines():        
        match = any(s.match(line) for s in md_re)
        if match:
            if len(current_chunk_lines) > 0:
                chunks.append(
                    Chunk(txt=' '.join(current_chunk_lines), meta={})
                )
            current_chunk_lines = []
        else:
            current_chunk_lines.append(line)

    if len(current_chunk_lines) > 0:
        chunks.append(
            Chunk(txt=' '.join(current_chunk_lines), meta={})
        )

    return chunks


def task_chunker(text: str):
    chunks: List[Chunk] = []

    return chunks

