#!/usr/bin/env python

import argparse
import pickle
import json
import re
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

    with open(args.output, 'wb') as new_file:
        pickle.dump(merged_list, new_file)

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

    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())
