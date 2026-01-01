#!/usr/bin/env python3

from embedder import best_match
from chunker import structure_aware

with open("test_doc.md", "r") as f:
    test_data = f.read()        

chunks = structure_aware(test_data.strip())
query = "what does h1 mean?" # "what restrictions are at block level?"
best_chunk = best_match(query, chunks)
print(f"Query:\n{query}")
print(f"Best Match:\n{chunks[best_chunk].txt}")

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

