#!/usr/bin/env python3

from embedder import best_match
from chunker import structure_aware, fixed_chunks
from semantic_chunker import semantic_chunks
import sys

if len(sys.argv) < 2:
    raise SystemExit("supply a prompt please")

with open("test_doc.md", "r") as f:
    test_data = f.read().strip()        

# chunks = structure_aware(test_data)
# chunks = fixed_chunks(test_data, 128, 16)
chunks = semantic_chunks(test_data)
query = sys.argv[1]

ranked = sorted(
    best_match(query, chunks),
    key=lambda x: x[1],
    reverse=True
)

print(f"Query:\n\t{query}")
print("Top rank results:")

for i, score in ranked[:5]:
    print("\t", score, chunks[i].txt)
