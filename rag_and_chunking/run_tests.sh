#!/usr/bin/env bash

run_test() {
	./inference_tests.py \
		--file "$2" \
		--prompt "$1" \
		--top-p 0.9  \
		--top-k 50 \
		--temperature 0.2
}

# generate prompt using RAG stuff first
echo "test 1"
PROMPT=./some_py.py
run_test PROMPT test_1
