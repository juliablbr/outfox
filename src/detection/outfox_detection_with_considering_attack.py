import asyncio
from pathlib import Path
import random

from src.utils import ROOT
random.seed(42)
from tqdm import tqdm
import time
import os
import argparse
from src.attacking.dipper_attacking import DipperParaphraser
from src.utils.utils import OutfoxDetectionLLM, parse_outfox_args, load_pkl, process_reply_from_chatgpt, make_mixed_data, generation_by_vllm, identify_attack_label, make_prompt_for_detection, make_prompt_for_attack, compute_metrics, save_metrics, save_predictions
from vllm.sampling_params import GuidedDecodingParams

def identify_attack_labels_of_lms_from_train(llm, lm_pss_train, train_ps_to_near_ps_human_lm_pairs_from_train):
    # Attack label includes "Good" or "Bad".
    # 'Good' example for attacker is an LLM-generated essay which our detector misclassify, retrieved from the training set.
    # 'Bad' example for attacker is an LLM-generated essay which our detector classify correctly, retrieved from the training set
    lm_and_attack_labels = []
    for ps, _, lm in lm_pss_train:
        lm_and_attack_labels.append(identify_attack_label(llm, ps, lm, train_ps_to_near_ps_human_lm_pairs_from_train))
    return lm_and_attack_labels


def create_attack(llm, train_ps, human_text, lm_text, train_ps_to_near_ps_human_lm_pairs_from_train, dp=None, attacking_method='outfox'):
    # considering our outfox attack
    if attacking_method == 'outfox':
        ps_human_lm_pairs_from_train = train_ps_to_near_ps_human_lm_pairs_from_train[train_ps]
        # detection to make in-context examples for outfox attacker
        lm_and_attack_labels = identify_attack_labels_of_lms_from_train(llm, ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train)
        prompt, human_essay_tokens = make_prompt_for_attack(lm_and_attack_labels, train_ps, human_text)
        attacked_essay = generation_by_vllm(prompt, llm, {
            "temperature": 1.3,
            "top_p": 1.0,
            "max_tokens": human_essay_tokens+50
        })[0]
    # considering DIPPER attack
    elif attacking_method == 'paraphrase':
        attacked_essay = dp.paraphrase(lm_text, lex_diversity=60, order_diversity=60, prefix='', do_sample=True, top_p=0.75, top_k=None, max_length=4096)

    return attacked_essay


def replace_some_llms_with_attacks(llm, example_for_detection, train_ps_to_near_ps_human_lm_pairs_from_train, dp=None, attacking_method='outfox'):
    # When considering attacks, we replace 3 LLM-generated essays out of retrieved 5 LLM-generated essays with attacked essays.
    indices = list(range(len(example_for_detection)))
    random.shuffle(indices)
    for i in indices[:3]:
        ps, human, lm = example_for_detection[i]
        attack = create_attack(llm, ps, human, lm, train_ps_to_near_ps_human_lm_pairs_from_train, dp, attacking_method)
        example_for_detection[i] = (ps, human, attack)
    return example_for_detection


def detection_by_chatgpt(llm, test_data, mixed_test_pss, test_ps_to_near_ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train, preds_path, dp=None, attacking_method='outfox'):
    prompts = []
    for text, ps in tqdm(zip(test_data, mixed_test_pss), total=len(test_data), desc=f'Creating prompts for outfox detection considering {attacking_method} attacks...'):
        example_for_detection = test_ps_to_near_ps_human_lm_pairs_from_train[ps][:5]
        example_for_detection_considering_attack = replace_some_llms_with_attacks(llm, example_for_detection, train_ps_to_near_ps_human_lm_pairs_from_train, dp, attacking_method)
        prompt = make_prompt_for_detection(text, example_for_detection_considering_attack)
        prompts.append(prompt)
    
    print("Starting detector")
    responses = generation_by_vllm(prompts, llm, {
        "guided_decoding": GuidedDecodingParams(choice=["LM", "Human"]),
        "temperature": 0,
        "top_p": 1.0
    })
    print("Finished detection")

    save_predictions(preds_path, responses)

    predictions = []
    for response in responses:
        prediction = process_reply_from_chatgpt(response)
        predictions.append(prediction)

    return predictions


def main():
    parser = parse_outfox_args()
    parser.add_argument('--attacking_method', help='Specify attacking methods.', required=True, choices=['outfox', 'paraphrase'])
    args = parser.parse_args()

    attacking_method = args.attacking_method
    model_name = args.model_name
    
    preds_folder = Path(ROOT / "results", "outfox_detection", model_name.replace(":", "_"))
    preds_folder.mkdir(parents=True, exist_ok=True)

    preds_path = preds_folder / f"with_considering_{attacking_method}_attack_{model_name}.csv"
    metrics_path = preds_folder / f"with_considering_{attacking_method}_attack_{model_name}_metrics.csv"
    
    test_data, test_labels, test_pss = make_mixed_data('data/common/test/test_humans.pkl', f'data/{model_name}/test/test_lms.pkl', 'data/common/test/test_problem_statements.pkl')

    dp = DipperParaphraser(0) if attacking_method == 'paraphrase' else None

    print(f"Start outfox detection considering {attacking_method} attacks for essays generated by {model_name}...")

    # Loading top-k (problem statement, human-written essay, LLM-generated essay) sets retrieved in advance based on the problem statement, using tf-idf.
    # Whatever LMs (FLAN, GPT-3.5) to be detected, our OUTFOX detector consider the essays by ChatGPT.
    test_ps_to_near_ps_human_lm_pairs_from_train = load_pkl('data/chatgpt/util/test_ps_to_near_ps_human_lm_pairs_from_train.pkl')
    train_ps_to_near_ps_human_lm_pairs_from_train = load_pkl('data/chatgpt/util/train_ps_to_near_ps_human_lm_pairs_from_train.pkl')

    llm = OutfoxDetectionLLM(args)
    preds = detection_by_chatgpt(llm, test_data, test_pss, test_ps_to_near_ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train, preds_path, dp, attacking_method)

    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(test_labels, preds)
    print(f"HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
    save_metrics(metrics_path, {
        "human_rec": human_rec,
        "machine_rec": machine_rec,
        "avg_rec": avg_rec,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    print(f"Saved metrics in `{metrics_path}`")
    

if __name__ == '__main__':
    main()
