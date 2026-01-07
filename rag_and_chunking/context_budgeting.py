#!/usr/bin/env python3

from typing import List, Tuple
import sys
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from chunker import Chunk
from tester import build_chunks, get_hybrid, load_test_data
from decoder_params.inference_tests import run_pixi_generate

bad_prompt = "what are there restrictions on?" # vague, open-ended
better_prompt = """
Answer the question using ONLY the context below.
If the answer is not explicitly stated, say "I don't know."

CONTEXT:
<rag_token>

QUESTION:
What restrictions are there?""" # convince the generator model to look at our context more specifically, with a better question


def top_n(n: int, scores: List[Tuple[int, float]], chunks: List[Chunk]) -> str:
    """
    returns a concatenated string containing the top N highest-scoring matches
    that can subsequently be added to a prompt for generation
    """
    top_scores = scores[:n]
    return ' '.join([chunks[idx].txt for idx, _ in top_scores])


def get_tokens(n: int, scores: List[Tuple[int, float]], chunks: List[Chunk]) -> str:
    """
    returns a concatenated string of 'n' tokens from our hybrid-scored and
    chunked data
    """
    rag_text = ""
    token_count = 0

    for idx, _ in scores:
        chunk_tokens = chunks[idx].txt.split(' ')
        diff = n - token_count
        if len(chunk_tokens) > diff:
            rag_text += ' '.join(chunk_tokens[:diff])
        else:
            rag_text += ' '.join(chunk_tokens)
        print(f"diff: {diff}, RT: {rag_text}\n---")
    return rag_text


def generate(query: str, rag_text: str, out_file: str) -> str:
    """
    takes the base query (including a token for where the rag input should go)
    then implants the RAG text before running it through the generation model
    """
    prompt = query.replace("<rag_token>", rag_text)
    print(f"prompting with '{prompt}'")
    report = run_pixi_generate(prompt=prompt, out_file=out_file) 
    return report.output or ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("supply a prompt please")
    test_data = load_test_data()
    query = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "test_doc.md"

    # exercise 1 - generate with fixed N chunks
    scores = get_hybrid(query, path)
    chunks = build_chunks(test_data)

    # top_5 = top_n(5, scores, chunks)
    # top_3 = top_n(3, scores, chunks)
    # top_1 = top_n(1, scores, chunks)

    # output_5 = generate(better_prompt, top_5, out_file="top_5_test")
    # output_3 = generate(better_prompt, top_3, out_file="top_3_test")
    # output_1 = generate(better_prompt, top_1, out_file="top_1_test")

    # exercise 2 - generate with fixed token budget
    fivetwelve = get_tokens(30, scores, chunks)
    # tentwentyfour = get_tokens(1024, scores, chunks)

    output_1 = generate(better_prompt, fivetwelve, out_file="top_1_test")
    # output_2 = generate(better_prompt, tentwentyfour, out_file="top_1_test")

    # exercise 3 - salience test

