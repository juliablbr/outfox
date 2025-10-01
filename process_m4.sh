#!/bin/sh
set -e

# OUTPUT_DIR="m4_pickles"
# M4_DIR="../../M4/data"
# SEED=42

# mkdir -p "$OUTPUT_DIR/individual"

# ./process_m4.py m4 --output "$OUTPUT_DIR/individual/wikipedia_%ss.pkl" $M4_DIR/wikipedia_chatgpt.jsonl \
# 	--problem_statement 'title "(.*)"' 'Write a encyclopedia entry about \1'

# ./process_m4.py m4 --output "$OUTPUT_DIR/individual/wikihow_%ss.pkl" $M4_DIR/wikihow_chatGPT.jsonl \
# 	--problem_statement "title 'How to (.*)'" 'Write a guide on how to \1'

# ./process_m4.py m4 --output "$OUTPUT_DIR/individual/reddit_%ss.pkl" $M4_DIR/reddit_chatGPT.jsonl \
# 	--problem_statement "Question: (.*)" 'Write an answer for the question: \1'

# ./process_m4.py m4 --output "$OUTPUT_DIR/individual/arxiv_%ss.pkl" $M4_DIR/arxiv_chatGPT.jsonl \
# 	--problem_statement "title: ([^.]+)" 'Write an abstract about the topic \1'

# ./process_m4.py merge --shuffle $SEED --split 94/3/3 "$OUTPUT_DIR/combined_problem_statements_%s.pkl" "$OUTPUT_DIR"/individual/*_problem_statements.pkl
# ./process_m4.py merge --shuffle $SEED --split 94/3/3 "$OUTPUT_DIR/combined_humans_%s.pkl" "$OUTPUT_DIR"/individual/*_humans.pkl
# ./process_m4.py merge --shuffle $SEED --split 94/3/3 "$OUTPUT_DIR/combined_lms_%s.pkl" "$OUTPUT_DIR"/individual/*_lms.pkl

OUTPUT_DIR="daigt_pickles"
DAIGT_DIR="../../DAIGT_essays"
SEED=42

mkdir -p "$OUTPUT_DIR/individual"

./process_m4.py daigt --output "$OUTPUT_DIR/individual/daigt_%ss.pkl" $DAIGT_DIR/daigt_external_dataset.csv 

./process_m4.py merge --shuffle $SEED --split 70/15/15 "$OUTPUT_DIR/daigt_problem_statements_%s.pkl" "$OUTPUT_DIR"/individual/*_problem_statements.pkl
./process_m4.py merge --shuffle $SEED --split 70/15/15 "$OUTPUT_DIR/daigt_humans_%s.pkl" "$OUTPUT_DIR"/individual/*_humans.pkl
./process_m4.py merge --shuffle $SEED --split 70/15/15 "$OUTPUT_DIR/daigt_lms_%s.pkl" "$OUTPUT_DIR"/individual/*_lms.pkl


