#!/usr/bin/env python3

import sys
import tester


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: hybrid_scores.py <query> [path]")
        return 1

    query = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "test_doc.md"

    scores = tester.get_hybrid(query, path)
    test_data = tester.load_test_data(path)
    chunks = tester.build_chunks(test_data)

    print("top 3 hybrid scores")
    for idx, score in scores[:3]:
        print(f"{idx}\t{score:.4f}\t{chunks[idx].txt[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
