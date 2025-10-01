#!/usr/bin/env python

import argparse
import pickle
import json
import csv
import re
import random
import pickle

class Processor:
    def __init__(self, problem_statement, human='human_text', lm='machine_text', prompt='prompt'):
        if problem_statement is not None:
            self.problem_statement = True
            self.problem_statement_regex = re.compile(problem_statement[0], re.M)
            self.problem_statement_template = problem_statement[1]
        else:
            self.problem_statement = False
        self.human_key = human
        self.lm_key = lm
        self.prompt_key = prompt

    def __call__(self, obj):
        return {
            'problem_statement': self.build_problem_statement(obj[self.prompt_key]),
            'human': obj[self.human_key],
            'lm': obj[self.lm_key],
        }

    def build_problem_statement(self, prompt):
        if self.problem_statement:
            match = self.problem_statement_regex.search(prompt)
            assert match is not None, f"unable to match regex to the prompt: {prompt}"
            return match.expand(self.problem_statement_template)
        else:
            return prompt

def process_m4_main(args: argparse.Namespace):
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

def process_daigt_main(args: argparse.Namespace):
    processor = Processor(args.problem_statement, human='text', lm='source_text', prompt='instructions')

    with open(args.input, 'r', newline='') as csv_:
        reader = csv.DictReader(csv_)
        data = [processor(row) for row in reader]

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

    data_split_num = list(map(int, args.split.split('/')))
    train_split = int((data_split_num[0] / sum(data_split_num)) * len(merged_list))
    val_split = int((data_split_num[1] / sum(data_split_num)) * len(merged_list))
    # test_split = (args.split[2] / args.split) * len(merged_list)

    # train-val-test split 
    train_list = merged_list[:train_split]
    val_list = merged_list[train_split:train_split+val_split]
    test_list = merged_list[train_split+val_split:]
    # print(len(train_list))
    # print(len(val_list))
    # print(len(test_list))

    with open(args.output.replace('%s', 'train'), 'wb') as train_file:
        pickle.dump(train_list, train_file)

    with open(args.output.replace('%s', 'val'), 'wb') as val_file:
        pickle.dump(val_list, val_file)

    with open(args.output.replace('%s', 'test'), 'wb') as test_file:
        pickle.dump(test_list, test_file)

def main(args: argparse.Namespace):
    if args.mode == 'm4':
        process_m4_main(args)
    elif args.mode == 'daigt':
        process_daigt_main(args)
    elif args.mode == 'merge':
        merge_main(args)
    else:
        print("try './process_m4.py --help'")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pre-processor for the m4 dataset")
    subparsers = parser.add_subparsers(dest='mode')

    process_m4_parser = subparsers.add_parser('m4', help="convert M4's .jsonl to .pkl")
    process_m4_parser.add_argument('input', type=str, help="path to the .jsonl file")
    process_m4_parser.add_argument('-o', '--output', type=str, help="if given, store the output as a pickle file rather than dumping to stdout")
    process_m4_parser.add_argument('-p', '--problem_statement', type=str, nargs=2, metavar=('REGEX', 'TEMPLATE'), help="regex template for the problem statement")

    process_daigt_parser = subparsers.add_parser('daigt', help="convert DAIGT's .csv to .pkl")
    process_daigt_parser.add_argument('input', type=str, help="path to the .csv file")
    process_daigt_parser.add_argument('-o', '--output', type=str, help="if given, store the output as a pickle file rather than dumping to stdout")
    process_daigt_parser.add_argument('-p', '--problem_statement', type=str, nargs=2, metavar=('REGEX', 'TEMPLATE'), help="regex template for the problem statement")

    merge_parser = subparsers.add_parser('merge', help="merges all your pickle into one big pickle")
    merge_parser.add_argument('output', type=str, help="output .pkl")
    merge_parser.add_argument('input', type=str, nargs='+', help="data/input .pkl")
    merge_parser.add_argument('--shuffle', type=int, metavar='SEED', help="shuffle the combined files with a given random seed")
    merge_parser.add_argument('--split', type=str, default='94/3/3', help="define test-val-test split, e.g 94/3/3")


    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())
