#!/usr/bin/env python

import argparse
import json
import re

class Processor:
    def __init__(self, problem_statement):
        self.problem_statement_regex = re.compile(problem_statement[0])
        self.problem_statement_template = problem_statement[1]

    def __call__(self, obj):
        return {
            'problem_statement': self.build_problem_statement(obj['prompt']),
            'human_text': obj['human_text'],
            'machine_text': obj['machine_text'],
        }

    def build_problem_statement(self, prompt):
        match = self.problem_statement_regex.search(prompt)
        assert match is not None, "unable to match regex to the prompt"
        return match.expand(self.problem_statement_template)

def process_main(args: argparse.Namespace):
    processor = Processor(args.problem_statement)

    with open(args.input, 'r') as jsonl:
        data = [processor(json.loads(line)) for line in jsonl]

    if args.output is None:
        for datum in data:
            print(json.dumps(datum))
    else:
        raise NotImplementedError

def merge_main(args: argparse.Namespace):
    pass

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
    merge_parser.add_argument('input', type=str, nargs='+', help="input .pkl")

    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())
