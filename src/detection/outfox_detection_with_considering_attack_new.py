from pathlib import Path
import random
from dataclasses import astuple


from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm

from src.attacking.dipper_attacking import DipperParaphraser
from src.utils.utils import OutfoxDetectionLLM, TaskTriplet, get_similar_triplets, combine_machine_text_label, parse_outfox_args, load_pkl, process_reply_from_chatgpt, make_mixed_data, generation_by_vllm, identify_attack_labels, make_prompt_for_detection, make_prompts_for_attacks, compute_metrics, save_metrics, save_predictions
from src.utils import ROOT


def split_examples_to_attack(
    similar_triplets_per_problem: list[list[TaskTriplet]],
    attack_n: int,
) -> tuple[list[list[TaskTriplet]], list[list[TaskTriplet]]]:
     # we will attack these indices
    random_triplet_indices = [
        random.sample(range(len(examples)), attack_n)
        for examples in similar_triplets_per_problem
    ]

    untouched = []
    to_be_attacked = []

    for similar_triplets, attack_indices in zip(similar_triplets_per_problem, random_triplet_indices):
        untouched_inner = []
        attack_inner = []

        for example_idx, triplet in enumerate(similar_triplets):
            if example_idx in attack_indices:
                attack_inner.append(triplet)
            else:
                untouched_inner.append(triplet)
        
        untouched.append(untouched_inner)
        to_be_attacked.append(attack_inner)

    return untouched, to_be_attacked


def dipper_attack(to_be_attacked: list[list[TaskTriplet]], dp) -> list[list[TaskTriplet]]:
    attacked = []
    # For each problem there are n examples. For each example, the machine-written text
    # is attacked.
    for examples in to_be_attacked:
        attacked_examples = []
        # For each example attack the machine-written text
        for example in examples:
            attacked_essay = dp.paraphrase(
                example.machine_text,
                lex_diversity=60,
                order_diversity=60,
                prefix='',
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_length=4096
            )
            attacked_examples.append(TaskTriplet(
                example.problem_statement,
                example.human_text,
                attacked_essay,
            ))
        attacked.append(attacked_examples)

    return attacked


def outfox_attack(
    llm,
    to_be_attacked: list[list[TaskTriplet]],
    train_ps_to_near_ps_human_lm_pairs_from_train
) -> list[list[TaskTriplet]]:
    # This will get the triplets with similar problem statements for each problem
    # statement in `to_be_attacked`
    similar_examples_to_attacked: list[list[list[TaskTriplet]]] = get_similar_triplets(
        to_be_attacked,
        train_ps_to_near_ps_human_lm_pairs_from_train,
        n_similar_triplets=10,
    )

    attack_labels = identify_attack_labels(
        llm,
        similar_examples_to_attacked,
        train_ps_to_near_ps_human_lm_pairs_from_train
    )

    attacker_examples = combine_machine_text_label(similar_examples_to_attacked, attack_labels)

    attacking_prompts, human_essay_tokens = make_prompts_for_attacks(
        to_be_attacked,
        attacker_examples,
    )

    print("Start generating essays.", len(attacking_prompts), "to be generated")
    # Generation of the attacked machine text
    attacked_essays = generation_by_vllm(attacking_prompts, llm, {
            "temperature": 1.3,
            "top_p": 1.0,
            "max_tokens": max(human_essay_tokens)+50
        }, "Generate essays")
    print("Finished generating essays")

    # Combining the original problem statement + human text + attacked machine text
    attacked_examples_per_problem = []
    attacked_essay_idx = 0
    for examples in to_be_attacked:
        attacked_examples = []
        for example in examples:
            attacked_text = attacked_essays[attacked_essay_idx]
            attacked_examples.append(
                TaskTriplet(example.problem_statement, example.human_text, attacked_text)
            )
            attacked_essay_idx += 1
        attacked_examples_per_problem.append(attacked_examples)
    
    return attacked_examples_per_problem


def detection_by_chatgpt(llm, test_data, test_pss, test_ps_to_near_ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train, preds_path, dp=None, attacking_method='outfox'):
    # This will contain for each test_essay/test_pss a list of 5 triplets
    # (problem statement, human text, machine text), that are related to the test_pss.
    # Where test_pss are the problem statements for each essay in the test_data
    similar_triplets_per_problem = get_similar_triplets(test_pss, test_ps_to_near_ps_human_lm_pairs_from_train)

    
    untouched, to_be_attacked = split_examples_to_attack(
        similar_triplets_per_problem,
        attack_n=3
    )
    if attacking_method == "outfox":
        attacked = outfox_attack(llm, to_be_attacked, train_ps_to_near_ps_human_lm_pairs_from_train)
    elif attacking_method == "dipper":
        attacked = dipper_attack(to_be_attacked, dp)
    else:
        raise ValueError(f"Unknown attacking method: {attacking_method}")
    
    examples_for_detector = [
        untoched_examples + attacked_examples
        for untoched_examples, attacked_examples in zip(untouched, attacked)
    ]

    # Create the prompts by using the examples from the training data where some
    # machine-written are replaced by attacked essays.
    prompts = []
    for target_text, example_for_detection in tqdm(
        zip(test_data, examples_for_detector),
        total=len(test_data),
        desc=f'Creating prompts for outfox detection considering {attacking_method} attacks...'
    ):
        pairs = [astuple(example) for example in example_for_detection]
        prompt = make_prompt_for_detection(target_text, pairs)
        prompts.append(prompt)


    # Run the detector for each prompt to detect whether the essay is Human or LM written 
    print("Starting detector")
    responses = generation_by_vllm(prompts, llm, {
        "guided_decoding": GuidedDecodingParams(choice=["LM", "Human"]),
        "temperature": 0,
        "top_p": 1.0
    }, "Target essay detection")
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

    random.seed(args.seed)

    attacking_method = args.attacking_method
    model_name = args.model_name
    detection_model = args.detection_model
    
    preds_folder = Path(ROOT / "results", "outfox_detection", detection_model, f"seed_{args.seed}")
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
