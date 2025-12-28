#!/usr/bin/env bash

run_test() {
	./inference_tests.py \
		--file "$5" \
		--prompt "$1" \
		--top-p "$2" \
		--top-k "$3" \
		--temperature "$4"
}

echo "eclipse test 1"
run_test "why do eclipses happen?" 1.0 1 0.0 eclipse_1

echo "eclipse test 2"
run_test "why do eclipses happen?" 0.9 50 0.2 eclipse_2

echo "eclipse test 3"
run_test "why do eclipses happen?" 0.95 100 0.9 eclipse_3

echo "---"
echo "haiku test 1"
run_test "write a haiku about a random historical figure" 1.0 1 0.0 haiku_1
echo "haiku test 2"
run_test "write a haiku about a random historical figure" 0.9 50 0.2 haiku_2
echo "haiku test 3"
run_test "write a haiku about a random historical figure" 0.95 100 0.9 haiku_3

echo "---"
echo "haiku test 2"
run_test "write a haiku about a random historical figure" 1.0 1 0.0 haiku_1
echo "haiku test 2"
run_test "write a haiku about a random historical figure" 0.9 50 0.2 haiku_2
echo "haiku test 3"
run_test "write a haiku about a random historical figure" 0.95 100 0.9 haiku_3

echo "---"
echo "haiku advanced test 1"
run_test "write a haiku about one historical figure by sampling from this list: [marc anthony, julius caesar, marcus aurelius, scipio africanus]" 1.0 1 0.0 haiku_1adv
echo "haiku advanced test 2"
run_test "write a haiku about one historical figure by sampling from this list: [marc anthony, julius caesar, marcus aurelius, scipio africanus]" 0.9 50 0.2 haiku_2adv
echo "haiku advanced test 3"
run_test "write a haiku about one historical figure by sampling from this list: [marc anthony, julius caesar, marcus aurelius, scipio africanus]" 0.95 100 0.9 haiku_3adv   
