import random
import sys
from src.utils import ROOT
from tqdm import tqdm
import time
import os
from pathlib import Path
from src.utils.utils import OutfoxDetectionLLM, parse_outfox_args, generation_by_vllm, load_pkl, process_reply_from_chatgpt, make_mixed_data, make_prompt_for_detection, compute_metrics, save_metrics, save_predictions, string2token_nums
from vllm.sampling_params import GuidedDecodingParams

def detection_by_chatgpt(test_data, mixed_test_pss, test_ps_to_near_ps_human_lm_pairs_from_train, preds_path, llm):
    prompts = []
    for text, ps in tqdm(zip(test_data, mixed_test_pss), total=len(test_data), desc='Creating prompts...'):
        prompt = make_prompt_for_detection(text, test_ps_to_near_ps_human_lm_pairs_from_train[ps][:5])
        prompts.append(prompt)

    print("Starting detector")
    responses = generation_by_vllm(prompts, llm, {
        "guided_decoding": GuidedDecodingParams(choice=["LM", "Human"]),
        "temperature": 0,
        "top_p": 1.0
    })
    print("Finished detection")

    save_predictions(preds_path, responses)
    print(f"Saved predictions in `{preds_path}`")

    predictions = []
    for response in responses:
        prediction = process_reply_from_chatgpt(response)
        predictions.append(prediction)

    return predictions



def main():
    args = parse_outfox_args().parse_args()

    if args.dataset == "outfox" and args.model_name is None:
        print("Using dataset `outfox` but missing --model_name")
        sys.exit()
    
    random.seed(args.seed)
    
    model_name = args.model_name
    detection_model = args.detection_model

    if args.dataset == "outfox":
        dataset_name = model_name

        test_human = "data/common/test/test_humans.pkl"
        test_lms = f"data/{dataset_name}/test/test_lms.pkl"
        test_problem_statements = "data/common/test/test_problem_statements.pkl"
        test_train_problem_statement_map = "data/chatgpt/util/test_ps_to_near_ps_human_lm_pairs_from_train.pkl"
    else:
        dataset_name = args.dataset

        dataset_folder = ROOT / f"{dataset_name}_pickles"
        test_human = dataset_folder / f"{'combined' if dataset_name == 'm4' else dataset_name}_humans_test.pkl"
        test_lms = dataset_folder / f"{'combined' if dataset_name == 'm4' else dataset_name}_lms_test.pkl"
        test_problem_statements = dataset_folder / f"{'combined' if dataset_name == 'm4' else dataset_name}_problem_statements_test.pkl"
        test_train_problem_statement_map = dataset_folder / "test_ps_to_near_ps_human_lm_pairs_from_train.pkl"

    
    
    preds_folder = Path(ROOT / "results", "outfox_detection", detection_model, f"seed_{args.seed}")
    preds_folder.mkdir(parents=True, exist_ok=True)
    preds_path = preds_folder / f"without_considering_attack_{dataset_name}.csv"
    metrics_path = preds_folder / f"without_considering_attack_{dataset_name}_metrics.csv"


    test_data, test_labels, mixed_test_pss = make_mixed_data(test_human, test_lms, test_problem_statements)
    
    print(f"Start outfox detection, without condiering attacks, for dataset {dataset_name}...")
    # Loading top-k (problem statement, human-written essay, LLM-generated essay) sets retrieved in advance based on the problem statement, using tf-idf.
    # Whatever LMs (FLAN, GPT-3.5) to be detected, our OUTFOX detector consider the essays by ChatGPT.
    test_ps_to_near_ps_human_lm_pairs_from_train = load_pkl(test_train_problem_statement_map)
    llm = OutfoxDetectionLLM(args)
    preds = detection_by_chatgpt(test_data, mixed_test_pss, test_ps_to_near_ps_human_lm_pairs_from_train, preds_path, llm)

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
