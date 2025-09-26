#!/bin/sh
set -e

OUTPUT_DIR="m4_pickles"
M4_DIR="../M4/data"
SEED=42

mkdir -p "$OUTPUT_DIR/individual"

./process_m4.py process --output "$OUTPUT_DIR/individual/wikipedia_%ss.pkl" $M4_DIR/wikipedia_chatgpt.jsonl \
	--problem_statement 'title "(.*)"' 'Write a encyclopedia entry about \1'

./process_m4.py process --output "$OUTPUT_DIR/individual/wikihow_%ss.pkl" $M4_DIR/wikihow_chatGPT.jsonl \
	--problem_statement "title 'How to (.*)'" 'Write a guide on how to \1'

./process_m4.py process --output "$OUTPUT_DIR/individual/reddit_%ss.pkl" $M4_DIR/reddit_chatGPT.jsonl \
	--problem_statement "Question: (.*)" 'Write an answer for the question: \1'

./process_m4.py process --output "$OUTPUT_DIR/individual/arxiv_%ss.pkl" $M4_DIR/arxiv_chatGPT.jsonl \
	--problem_statement "title: ([^.]+)" 'Write an abstract about the topic \1'

./process_m4.py merge --shuffle $SEED "$OUTPUT_DIR/combined_problem_statements_%s.pkl" "$OUTPUT_DIR"/individual/*_problem_statements.pkl
./process_m4.py merge --shuffle $SEED "$OUTPUT_DIR/combined_humans_%s.pkl" "$OUTPUT_DIR"/individual/*_humans.pkl
./process_m4.py merge --shuffle $SEED "$OUTPUT_DIR/combined_lms_%s.pkl" "$OUTPUT_DIR"/individual/*_lms.pkl
