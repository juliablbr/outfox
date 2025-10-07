from dataclasses import astuple, dataclass
from typing import Any, Optional
import pickle
import json
import random
from tqdm import tqdm
from functools import partial

import torch
import tiktoken
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pydantic import BaseModel
import csv
import argparse
from pathlib import Path
#
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pprint import pprint
#

print('Loading tiktoken tokenizer...')
gpt_turbo_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def parse_outfox_args():
    parser = argparse.ArgumentParser(description='Outfox model detection parser')
    parser.add_argument("--dataset", help="The dataset to be used", default="outfox", choices=["outfox", "daigt", "m4"])
    parser.add_argument('--model_name', help='Specify target llms to be detected.', choices=['chatgpt', 'flan_t5_xxl', 'text_davinci_003'])
    parser.add_argument("--detection_model", type=str, default="google/gemma-3-270m", help="The model name of the detection model")
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float)
    parser.add_argument("--max_num_seqs", default=256, type=int)
    parser.add_argument("--max_model_len", default=None, type=int)
    parser.add_argument("--max-tokens", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    return parser


@dataclass
class TaskTriplet:
    problem_statement: str
    human_text: str
    machine_text: str


@dataclass
class AttackTriplet:
    problem_statement: str
    machine_text: str
    detector_label: str


def save_predictions(path: Path, predictions):
    with path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "response"])
        for idx, response in enumerate(predictions):
            writer.writerow([idx, response])

def save_metrics(path: Path, metrics: dict[str, float]):
    with path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric_name", "metric_value"])
        for name, value in metrics.items():
            writer.writerow([name, value])

class OutfoxDetectionLLM:
    def __init__(self, args):
        self.llm = LLM(model=args.detection_model,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       max_num_seqs=args.max_num_seqs,
                       max_model_len=args.max_model_len,
                       seed=args.seed)
        #
        self.args = args
        print(f"\nLoaded the {args.detection_model} model")

    def generate(self, prompts, additional_sampling_params: dict[str, Any], use_tqdm = None):
        sampling_params = SamplingParams(seed=self.args.seed)
        for k, v in additional_sampling_params.items():
            if hasattr(sampling_params, k) and additional_sampling_params[k] is not None:
                setattr(sampling_params, k, v)
        return self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)

class LLMAnswer(BaseModel):
    detection: str


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def json2dict(path):
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_model_and_tokenizer(name):
    base_model = T5ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, offload_folder="offload")
    base_tokenizer = T5Tokenizer.from_pretrained(name)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer


def truncate_text(text, truncated_length, truncate_type):
    if truncate_type == 'token':
        truncated_encodes = gpt_turbo_encoding.encode(text, disallowed_special=())[:truncated_length]
        return gpt_turbo_encoding.decode(truncated_encodes)
    elif truncate_type == 'word':
        return ' '.join(text.split(' ')[:truncated_length])


def string2token_nums(string):
    num_tokens = 0
    num_tokens = len(gpt_turbo_encoding.encode(string, disallowed_special=()))
    return num_tokens


def make_mixed_data(human_path, lm_path, ps_path):
    random.seed(42)
    humans, lms, pss = load_pkl(human_path), load_pkl(lm_path), load_pkl(ps_path)
    humans_with_label_ps, lms_with_label_ps = [(human, '0', ps) for human, ps in zip(humans, pss)], [(lm, '1', ps) for lm, ps in zip(lms, pss)]
    all_with_label_ps = humans_with_label_ps + lms_with_label_ps
    random.shuffle(all_with_label_ps)
    data = [t[0] for t in all_with_label_ps]
    labels = [t[1] for t in all_with_label_ps]
    pss = [t[2] for t in all_with_label_ps]
    return (data, labels, pss)  


def generation_by_vllm(
    prompts,
    llm,
    sampling_params={},
    tqdm_description: Optional[str] = None
):
    use_tqdm = True
    if tqdm_description is not None:
        use_tqdm = partial(tqdm, desc=tqdm_description, leave=True)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)

    return results

# Create a prompt for the detector to classify the `target_text` by using examples (pairs)
def make_prompt_for_detection(target_text, pairs):
    instruction = 'Please classify whether the text is generated by a Human or a Language Model(LM).\n'
    assert len(pairs) == 5, 'Incorrect example nums'
    prompt = instruction
    ### Truncating text depending on the maximum length.
    target_text_tokens = string2token_nums(target_text)
    if target_text_tokens > 2000:
        target_text = truncate_text(target_text, 2000, 'token')

    truncated_length = (4000 - (string2token_nums(instruction) + (string2token_nums(f'Text: Answer: Human\n') + string2token_nums(f'Text: Answer: LM\n')) * 5 + string2token_nums(f'Text: {target_text} Answer: '))) // 10
    all_data = []
    for pair in pairs:
        human, lm = pair[-2], pair[-1]
        human = truncate_text(human, truncated_length, 'token')
        lm = truncate_text(lm, truncated_length, 'token')
        all_data.append((human, 'Human'))
        all_data.append((lm, 'LM'))
    random.shuffle(all_data)
    for essay, label in all_data:
        prompt += f'Text: {essay} Answer: {label}\n'
    prompt += f'Text: {target_text} Answer: '
    return prompt


def make_prompts_for_detections(
    examples: list[list[list[TaskTriplet]]],
    train_ps_to_near_ps_human_lm_pairs_from_train,
) -> list[str]:
    similar_examples: list[list[list[list[TaskTriplet]]]] = get_similar_triplets(
        examples,
        train_ps_to_near_ps_human_lm_pairs_from_train,
        n_similar_triplets=5,
    )

    prompts = []
    for (examples_, similar_examples_) in zip(examples, similar_examples):
        for (inner_examples, inner_similar_examples) in zip(examples_, similar_examples_):
            for target, detection_examples in zip(inner_examples, inner_similar_examples):
                pairs = [astuple(detection_example) for detection_example in detection_examples]
                prompt = make_prompt_for_detection(target.machine_text, pairs)
                prompts.append(prompt)
    
    return prompts


def make_prompt_for_attack(lm_and_attack_labels, problem_statement, human_essay):
    human_essay_tokens = string2token_nums(human_essay)
    human_essay = truncate_text(human_essay, 2000, "token")
    human_essay_words = len(human_essay.split(' '))

    instruction_1 = f'Here are the results of detecting whether each essay from each problem statement is generated by a Human or a Language Model(LM).\n'
    instruction_2 = f'\nGiven the following problem statement, please write an essay detected as Human in {human_essay_words} words with a clear opinion.\n\nProblem statement:\n{problem_statement}\nAnswer:\nHuman\nEssay:\n'
    prompt = instruction_1

    all_ps_length = sum([max(string2token_nums(ps), 1000) for ps, _, _ in lm_and_attack_labels])
    truncated_length = (4000 - (string2token_nums(instruction_1 + instruction_2) + (string2token_nums(f'Problem statement:\n\nAnswer:\nHuman\nEssay:\n\n') + string2token_nums(f'Problem statement:\n\nAnswer:\nLM\nEssay:\n\n')) * 5 + all_ps_length + human_essay_tokens + 50)) // 10
    assert len(lm_and_attack_labels) == 10, 'Invalid test data size'

    for ps, lm_essay, attack_label in lm_and_attack_labels:
        ps = truncate_text(ps, truncated_length // 3, "token")
        lm_essay = truncate_text(lm_essay, truncated_length // 3 * 2, 'token')
        prompt += f"Problem statement:\n{ps}\nAnswer:\n{'Human' if attack_label == 'Good' else 'LM'}\nEssay:\n{lm_essay}\n"
    # So the idea is that the attacker, gets a prompt with similar texts to the problem statement, with the info on how the detector classified them
    # That way the attacker, can try and generate a text that looks as if it was human written, based on the previously made assesment of the detector
    prompt += instruction_2
    return (prompt, human_essay_tokens)


def make_prompts_for_attacks(
    triplets: list[list[TaskTriplet]],
    attacker_examples_per_problem: list[list[list[AttackTriplet]]]
) -> tuple[list[str], list[int]]:
    prompts = []
    human_essay_tokens_length = []

    for examples, attacker_examples_ in zip(triplets, attacker_examples_per_problem):
        for example, attacker_examples in zip(examples, attacker_examples_):
            # Converting the list[AttackTriplet] to a list[tuple[str, str, str]] for
            # make_prompt_for_attack
            lm_and_attack_labels = [
                astuple(attacker_example)
                for attacker_example in attacker_examples
            ]
            prompt, human_essay_tokens = make_prompt_for_attack(
                lm_and_attack_labels,
                example.problem_statement,
                example.human_text,
            )
            prompts.append(prompt)
            human_essay_tokens_length.append(human_essay_tokens)
    
    return prompts, human_essay_tokens_length


def identify_attack_label(llm, ps, lm, ps_human_lm_train):
    # Attack label includes "Good" or "Bad".
    # 'Good' example for attacker is an LLM-generated essay which our detector misclassify, retrieved from the training set.
    # 'Bad' example for attacker is an LLM-generated essay which our detector classify correctly, retrieved from the training set
    # ps_human_lm_train, is a list of tuples, that contain similar problem statements, with their respective lm-text and human-text
    # Remove the problem statement out of the tuple, so we can throw it into make_prompt_for_detection which does not accept a tuple of 3, but only 2
    human_lm_train = [
        (human_text, lm_text)
        for _, human_text, lm_text in ps_human_lm_train
    ]
    # Create a prompt, to classify the lm (machine-written) text we want to attack, this
    # prompt will use the nearby (human, lm) texts as examples to classify it
    prompt = make_prompt_for_detection(lm, human_lm_train[:5])
    # get the answer from the detector for the LM written text, if the text is a human or lm text
    responses = generation_by_vllm(prompt, llm, {
        "guided_decoding": GuidedDecodingParams(choice=["LM", "Human"]),
        "temperature": 0,
        "top_p": 1.0
    })
    # See if the LM properly classified the text (if it did it correctly or not), since
    # it always is a LM text it has to classify it as LM.
    pred = [process_reply_from_chatgpt(response) for response in responses]
    # The attack label will be "Good", is the LM wrongly classifies the text, if it
    # correctly classifies the text, it is considered label Bad.
    # pred == 0 implies it was classifed as human
    attack_label = "Good" if pred == "0" else "Bad"
    return (ps, lm, attack_label)


def identify_attack_labels(
    llm,
    similar_examples: list[list[list[TaskTriplet]]],
    train_ps_to_near_ps_human_lm_pairs_from_train,
) -> list[str]:
    detection_prompts = make_prompts_for_detections(
        similar_examples,
        train_ps_to_near_ps_human_lm_pairs_from_train
    )
    
    print("Start Identify attack labels")
    responses = generation_by_vllm(detection_prompts, llm, {
        "guided_decoding": GuidedDecodingParams(choice=["LM", "Human"]),
        "temperature": 0,
        "top_p": 1.0
    }, "Identify attack labels")
    print("Finished Identify attack labels")

    attack_labels = [
        "Good" if process_reply_from_chatgpt(response) == "0" else "Bad"
        for response in responses
    ]
    return attack_labels


def combine_machine_text_label(
    similar_examples: list[list[list[TaskTriplet]]],
    labels: list[str],
) -> list[list[list[AttackTriplet]]]:
    assert (
        sum(
            len(examples)
            for inner_similar_examples in similar_examples
            for examples in inner_similar_examples
        )
    ) == len(labels)

    label_idx = 0

    batched_attacker_examples = []
    for similar in similar_examples:
        attacker_examples_per_attack = []

        for examples in similar:
            attack_examples = []

            for example in examples:
                attack_examples.append(AttackTriplet(
                    example.problem_statement,
                    example.machine_text,
                    labels[label_idx]
                ))
                label_idx += 1

            attacker_examples_per_attack.append(attack_examples)
        batched_attacker_examples.append(attacker_examples_per_attack)
    
    return batched_attacker_examples




def process_reply_from_chatgpt(res):
    # If the reply from the LLM is None, the label will be human-written
    if res is None:
        return "0"
    elif res.lower() == "human":
        return "0"
    elif res.lower() == "language model" or res.lower() == "lm":
        return "1"
    # If the label is neither recognised as "human" or "lm" it will be labeled as human
    else:
        print(f"[warn] received unknown reply: {res}")
        return "0"


def compute_three_recalls(labels, preds):
    all_n, all_p, tn, tp = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == '0':
            all_p += 1
        if label == '1':
            all_n += 1
        if label == pred == '0':
            tp += 1 
        if label == pred == '1':
            tn += 1
    human_rec = tp * 100 / all_p
    machine_rec = tn * 100 / all_n
    avg_rec = (human_rec + machine_rec) / 2
    return (human_rec, machine_rec, avg_rec)


def compute_metrics(labels, preds):
    human_rec, machine_rec, avg_rec = compute_three_recalls(labels, preds)
    acc, precision, recall, f1 = accuracy_score(labels, preds), precision_score(labels, preds, pos_label='1'), recall_score(labels, preds, pos_label='1'), f1_score(labels, preds, pos_label='1')
    return (human_rec, machine_rec, avg_rec, acc, precision, recall, f1)


def get_similar_triplets(
    problem_statement,
    problem_statement_mapping,
    n_similar_triplets: int = 5
):
    """
    Recursively get similar triplets for nested problem statement structures.
    Returns structure matching input depth: 
    - `str` -> `list[TaskTriplet]`
    - `TaskTriplet` -> `list[TaskTriplet]`
    - `list[str]` -> `list[list[TaskTriplet]]`
    - `list[TaskTriplet]` -> `list[list[TaskTriplet]]`
    - `list[list[str]]` -> `list[list[list[TaskTriplet]]]`
    - `list[list[TaskTriplet]]` -> `list[list[list[TaskTriplet]]]`
    etc.
    """

    # Base case: single problem statement (string)
    if isinstance(problem_statement, str):
        return [
            TaskTriplet(*triplet)
            for triplet in problem_statement_mapping[problem_statement][:n_similar_triplets]
        ]
    
    # Base case: single TaskTriplet object
    elif isinstance(problem_statement, TaskTriplet):
        return [
            TaskTriplet(*triplet)
            for triplet in problem_statement_mapping[problem_statement.problem_statement][:n_similar_triplets]
        ]
    
    # Recursive case: list of problem statements or TaskTriplets (or nested lists)
    elif isinstance(problem_statement, list):
        return [
            get_similar_triplets(item, problem_statement_mapping, n_similar_triplets)
            for item in problem_statement
        ]
    
    else:
        raise TypeError(f"Expected str, TaskTriplet, or list, got {type(problem_statement)}")
