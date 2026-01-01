#!/usr/bin/env python3
"""
Chunking utilities for Markdown documents, geared for RAG pipelines.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    text: str
    meta: dict


def read_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_whitespace(text: str) -> str:
    # Collapse repeated whitespace for more stable chunk sizes.
    return re.sub(r"\s+", " ", text).strip()


def chunk_by_headings(markdown: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_heading = None
    current_lines: List[str] = []

    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if current_lines:
                chunks.append(
                    Chunk(
                        text="\n".join(current_lines).strip(),
                        meta={"heading": current_heading},
                    )
                )
                current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append(
            Chunk(text="\n".join(current_lines).strip(), meta={"heading": current_heading})
        )
    return [c for c in chunks if c.text]


def chunk_by_paragraphs(markdown: str) -> List[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", markdown) if p.strip()]
    return [Chunk(text=p, meta={"type": "paragraph"}) for p in paragraphs]


def chunk_by_sentences(markdown: str) -> List[Chunk]:
    text = normalize_whitespace(markdown)
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    return [Chunk(text=s, meta={"type": "sentence"}) for s in sentences]


def _sliding_window(tokens: List[str], size: int, overlap: int) -> Iterator[List[str]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        yield tokens[start:end]
        if end == len(tokens):
            break
        start = end - overlap


def chunk_by_words(markdown: str, size: int, overlap: int = 0) -> List[Chunk]:
    tokens = normalize_whitespace(markdown).split(" ")
    chunks = []
    for window in _sliding_window(tokens, size, overlap):
        chunks.append(Chunk(text=" ".join(window), meta={"type": "word_window"}))
    return chunks


def chunk_by_chars(markdown: str, size: int, overlap: int = 0) -> List[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")
    text = markdown
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(text=text[start:end], meta={"type": "char_window"}))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_by_lines(markdown: str, size: int, overlap: int = 0) -> List[Chunk]:
    lines = markdown.splitlines()
    chunks = []
    for window in _sliding_window(lines, size, overlap):
        chunks.append(Chunk(text="\n".join(window), meta={"type": "line_window"}))
    return chunks


def _write_chunks(chunks: Iterable[Chunk]) -> None:
    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n--- chunk {idx} ---")
        if chunk.meta:
            print(f"meta: {chunk.meta}")
        print(chunk.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Markdown for RAG usage.")
    parser.add_argument("path", help="Path to the Markdown file")
    parser.add_argument(
        "--mode",
        choices=["headings", "paragraphs", "sentences", "words", "chars", "lines"],
        default="paragraphs",
        help="Chunking strategy",
    )
    parser.add_argument("--size", type=int, default=200, help="Window size")
    parser.add_argument("--overlap", type=int, default=0, help="Window overlap")
    args = parser.parse_args()

    markdown = read_markdown(args.path)

    if args.mode == "headings":
        chunks = chunk_by_headings(markdown)
    elif args.mode == "paragraphs":
        chunks = chunk_by_paragraphs(markdown)
    elif args.mode == "sentences":
        chunks = chunk_by_sentences(markdown)
    elif args.mode == "words":
        chunks = chunk_by_words(markdown, args.size, args.overlap)
    elif args.mode == "chars":
        chunks = chunk_by_chars(markdown, args.size, args.overlap)
    elif args.mode == "lines":
        chunks = chunk_by_lines(markdown, args.size, args.overlap)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    _write_chunks(chunks)


if __name__ == "__main__":
    main()
