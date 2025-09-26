#!/usr/bin/env python

import argparse
import pickle
import json
import re
import random
import pickle

class Processor:
    def __init__(self, problem_statement):
        self.problem_statement_regex = re.compile(problem_statement[0])
        self.problem_statement_template = problem_statement[1]

    def __call__(self, obj):
        return {
            'problem_statement': self.build_problem_statement(obj['prompt']),
            'human': obj['human_text'],
            'lm': obj['machine_text'],
        }

    def build_problem_statement(self, prompt):
        match = self.problem_statement_regex.search(prompt)
        assert match is not None, f"unable to match regex to the prompt: {prompt}"
        return match.expand(self.problem_statement_template)

def process_main(args: argparse.Namespace):
    processor = Processor(args.problem_statement)

    with open(args.input, 'r') as jsonl:
        data = [processor(json.loads(line)) for line in jsonl]

    if args.output is None:
        for datum in data:
            print(json.dumps(datum))
    else:
        assert '%s' in args.output, "no '%s' in output path"
        assert len(data) > 0, "no data"

        for key in data[0].keys():
            with open(args.output.replace('%s', key), 'wb') as pkl:
                pickle.dump([datum[key] for datum in data], pkl)

def merge_main(args: argparse.Namespace):
    merged_list = []
    for path in args.input: 
        with open(path, 'rb') as file: 
            loaded_data = pickle.load(file)
            merged_list += loaded_data

    if args.shuffle is not None:
        random.seed(args.shuffle)
        random.shuffle(merged_list)

    train_split = int((args.split[0] / sum(args.split)) * len(merged_list))
    val_split = int((args.split[1] / args.split) * len(merged_list))
    # test_split = (args.split[2] / args.split) * len(merged_list)

    # train-val-test split 
    train_list = merged_list[:train_split]
    val_list = merged_list[train_split:train_split+val_split]
    test_list = merged_list[train_split+val_split:]

    with open(args.output.replace('%s', 'train'), 'wb') as train_file:
        pickle.dump(train_list, train_file)

    with open(args.output.replace('%s', 'val'), 'wb') as val_file:
        pickle.dump(val_list, val_file)

    with open(args.output.replace('%s', 'test'), 'wb') as test_file:
        pickle.dump(test_list, test_file)

def main(args: argparse.Namespace):
    if args.mode == 'process':
        process_main(args)
    elif args.mode == 'merge':
        merge_main(args)
    else:
        print("try './process_m4.py --help'")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pre-processor for the m4 dataset")
    subparsers = parser.add_subparsers(dest='mode')

    process_parser = subparsers.add_parser('process', help='convert .json to .pkl')
    process_parser.add_argument('input', type=str, help="path to the .jsonl file")
    process_parser.add_argument('-o', '--output', type=str, help="if given, store the output as a pickle file rather than dumping to stdout")
    process_parser.add_argument('-p', '--problem_statement', type=str, required=True, nargs=2, metavar=('REGEX', 'TEMPLATE'), help="regex template for the problem statement")

    merge_parser = subparsers.add_parser('merge', help="merges all your pickle into one big pickle")
    merge_parser.add_argument('output', type=str, help="output .pkl")
    merge_parser.add_argument('input', type=str, nargs='+', help="data/input .pkl")
    merge_parser.add_argument('--shuffle', type=int, metavar='SEED', help="shuffle the combined files with a given random seed")
    merge_parser.add_argument('--split', type=list, help="list of test-val-test split, e.g [94,3,3]")


    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())
