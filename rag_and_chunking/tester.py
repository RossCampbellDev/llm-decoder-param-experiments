#!/usr/bin/env python3

from embedder import best_match
from chunker import structure_aware
import sys

if len(sys.argv) < 2:
    raise SystemExit("supply a prompt please")

with open("test_doc.md", "r") as f:
    test_data = f.read()        

chunks = structure_aware(test_data.strip())
query = sys.argv[1] # "what restrictions are at block level?"
# best_chunk = best_match(query, chunks)
# print(f"Query:\n{query}")
# print(f"Best Match:\n{chunks[best_chunk].txt}")

ranked = sorted(
    best_match(query, chunks),
    key=lambda x: x[1],
    reverse=True
)

print(f"Query:\n\t{query}")
print(f"Top rank results:")

for i, score in ranked[:5]:
    print("\t", score, chunks[i].txt[:150])

# for chunk in fixed_chunks(test_data.strip(), 512, 26):
#     print("-"*20)
#     print(chunk.text.replace("\n", ""))

# print(f"Query: {query}")
# print(f"Best Matching Chunk:\n{best_chunk}")
 
# query = "what time is it?"
 
# targets = [
#     "the hour is late",
#     "twelve kilometres per hour",
#     "the aardvark is a large animal",
#     "it is noon, mid-day",
# ]
 
# print(best_match(query, targets))

