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
from src.utils.utils import OutfoxDetectionLLM, TaskTriplet, get_similar_triplets, parse_outfox_args, load_pkl, process_reply_from_chatgpt, make_mixed_data, generation_by_vllm, identify_attack_label, make_prompt_for_detection, make_prompt_for_attack, compute_metrics, save_metrics, save_predictions
from vllm.sampling_params import GuidedDecodingParams


def identify_attack_labels_of_lms_from_train(llm, lm_pss_train, train_ps_to_near_ps_human_lm_pairs_from_train):
    # Attack label includes "Good" or "Bad".
    # 'Good' example for attacker is an LLM-generated essay which our detector misclassify, retrieved from the training set.
    # 'Bad' example for attacker is an LLM-generated essay which our detector classify correctly, retrieved from the training set
    lm_and_attack_labels = []
    # So for each of the problem statement, that is similar to the train problem statement. we see if it gets classified good/bad by the detector, where good is a poorly classified example
    for ps, _, lm in lm_pss_train: # lm_pss_train, is a list with tuples of (ps, human-text, lm-text) these tuples are similar problem statements to the train problem statement
        # Identify whether or not, the language model can properly say if the text is machine generated
        # Returns a tuple in the sense of (problem statement, lm-text, good/bad), where good implies it DID NOT properly classify the text (so wrongly)
        ps_human_lm_train = train_ps_to_near_ps_human_lm_pairs_from_train[ps] # This gives a list in the sense of, [(ps, human lm), (ps, human, lm)] of similar problem statements, which is given by indexing the current ps
        lm_and_attack_labels.append(identify_attack_label(llm, ps, lm, ps_human_lm_train))
    return lm_and_attack_labels

def create_attack(llm, train_ps, human_text, lm_text, train_ps_to_near_ps_human_lm_pairs_from_train, dp=None, attacking_method='outfox'):
    # considering our outfox attack
    if attacking_method == 'outfox':
        # Grab all similar problem statement/text pairs, to the current train_ps (which is the ps that we will be attacking)
        ps_human_lm_pairs_from_train = train_ps_to_near_ps_human_lm_pairs_from_train[train_ps]
        # detection to make in-context examples for outfox attacker
        # identify_attack_labels_of_lms_from_train, sees how the detector would have classified,
        # it does this for every problem statement-text pair, that is similar to the current train_ps
        lm_and_attack_labels = identify_attack_labels_of_lms_from_train(llm, ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train)
        # Create a prompt, to properly attack the current train_ps
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
    # When considering attacks, we replace 3 LLM-generated essays out of retrieved 5
    # LLM-generated essays with attacked essays.
    indices = list(range(len(example_for_detection)))
    random.shuffle(indices)
    #
    for i in indices[:3]:
        problem_statement, human_essay, lm_essay = example_for_detection[i]
        # Attack a machine-generated essay
        attacked_essay = create_attack(
            llm,
            problem_statement,
            human_essay,
            lm_essay,
            train_ps_to_near_ps_human_lm_pairs_from_train,
            dp,
            attacking_method
        )
        # Overwrite the example with the same problem statement, human-written essay and
        # the newly created attack essay
        example_for_detection[i] = (problem_statement, human_essay, attacked_essay)
    return example_for_detection


def detection_by_chatgpt(llm, test_data, test_pss, test_ps_to_near_ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train, preds_path, dp=None, attacking_method='outfox'):
    # This will contain for each test_essay/test_pss a list of 5 triplets
    # (problem statement, human text, machine text), that are related to the test_pss.
    # Where test_pss are the problem statements for each essay in the test_data
    similar_triplets_per_problem = get_similar_triplets(test_pss, test_ps_to_near_ps_human_lm_pairs_from_train)

    # For each 5 examples, we replace (attack) 3 of them. always 3, since outfox does this too
    # Therefore, we now grab the 3 examples we want to attack at random for each datapoint
    random_triplet_indices = [random.sample(range(5), 3) for _ in range(len(similar_triplets_per_problem))] # we will attack these indices
    to_detect_prompts = [] # The prompts related to the random_triplet_indices, so it is a list of length test_data, and each element has 3 prompts for each of the 3 indices we want to attack
    random_triplets = []
    for similar_triplets, indices in zip(similar_triplets_per_problem, random_triplet_indices):
        triplets_to_attack = [similar_triplets[idx] for idx in indices]
        triplet_problem_statements = [triplet.problem_statement for triplet in triplets_to_attack]
        similar_triplets_per_train_problem = get_similar_triplets(triplet_problem_statements, train_ps_to_near_ps_human_lm_pairs_from_train) # list of size 3, with lists of size 5 that each contain the 5 similar examples for a triplet we want to attack
        #
        temp_prompts = [] # List of len(3), with a list of 5 prompts for each index
        for triplet, triplet_similar in zip(triplets_to_attack, similar_triplets_per_train_problem):
            #
            temp_triplet_problem_statements = [triplet.problem_statement for temp_triplet in triplet_similar]
            similar_triplets_per_train_problems_triplets = get_similar_triplets(temp_triplet_problem_statements, train_ps_to_near_ps_human_lm_pairs_from_train)
            #
            triplet_similar_pairs = [(triplet.human_text, triplet.machine_text) for triplet in triplet_similar] # make_prompt_for_detection, needs just a pair of humand and lm text, no PS
            triplet_prompt = make_prompt_for_detection(triplet.machine_text, triplet_similar_pairs) # The prompt for this training example, to classify it
            temp_prompts.append(triplet_prompt) # append, this example its prompt, to later append a list of 3 prompts to to_detect_prompts
        #
        to_detect_prompts.append(temp_prompts) # List of len(test_data) with each index containing a list of len(3) where each index is a list of len(5) containing 5 prompts
        random_triplets.append(triplets_to_attack) # The three triplets we decided to attack
    
    # So we have to_detect_prompts, which is of len test_data and contains 3 lists of 5 prompts
    # But to generate it as one batch, we will turn it into one large list of prompts that is of len(5*3*len(test_data))
    # expand the to_detect_prompts to be of size len(5*3*len(test_data))
    flattened_to_detect_prompts = [item for sublist1 in to_detect_prompts for sublist2 in sublist1 for item in sublist2] # get the answer from the detector for all of the various prompts we had to deal with
    responses = generation_by_vllm(flattened_to_detect_prompts, llm, {
        "guided_decoding": GuidedDecodingParams(choice=["LM", "Human"]),
        "temperature": 0,
        "top_p": 1.0
    })
    # The attack label will be "Good", is the LM wrongly classifies the text, if it
    # correctly classifies the text, it is considered label Bad.
    # pred == 0 implies it was classifed as human
    flat_attack_labels = [ # The labels of the chosen indices
        "Good" if process_reply_from_chatgpt(response) == 0 else "Bad"
        for response in responses
    ]
    attack_labels = [ # Combine it again to be of len test_data, where it is lists of size 3, which each have a list of 5 as their element
        [flat_attack_labels[i+j*5:i+(j+1)*5] for j in range(3)]
        for i in range(0, len(flat_attack_labels), 15)
    ]
    # problem_statements,  machine_generated_texts, attack_labels
    to_attack_prompts = []
    # Create a prompt, to properly attack the current train_ps
    for three_attack_labels, three_triplets in zip(attack_labels, random_triplets):
        for labels, triplet in zip(three_attack_labels, three_triplets):
            # ps, lm_essay, attack_label
            prompt, human_essay_tokens = make_prompt_for_attack(labels, triplet.problem_statement, triplet.human_text)




    untouched, to_be_attacked = split_examples_to_attack(
        similar_triplets_per_problem,
        attack_n=3
    )
    if attacking_method == "outfox":
        attacked = outfox_attack(llm, to_be_attacked, ...)
    elif attacking_method == "dipper":
        attacked = dipper_attack(to_be_attacked, ...)
    else:
        raise ValueError(f"Unknown attacking method: {attacking_method}")
    
    examples_for_detector = [
        untoched_examples.extend(attacked_examples)
        for untoched_examples, attacked_examples in zip(untouched, attacked)
    ]

    # example_pairs_for_detector = []
    # for target_text, problem_statement in tqdm(
    #     zip(test_data, mixed_test_pss),
    #     total=len(test_data),
    #     desc=f'Creating examples for outfox detector considering {attacking_method} attacks...'
    # ):
    #     # Get 5 essay pairs (human-written, machine-written) from the train set, where
    #     # the problem statement of each pair is close/similar to the test problem
    #     # statement.
    #     example_for_detection = test_ps_to_near_ps_human_lm_pairs_from_train[problem_statement][:5]

    #     example_for_detection_considering_attack = replace_some_llms_with_attacks(
    #         llm,
    #         example_for_detection,
    #         train_ps_to_near_ps_human_lm_pairs_from_train,
    #         dp,
    #         attacking_method
    #     )

    #     example_pairs_for_detector.append(example_for_detection_considering_attack)

    # Create the prompts by using the examples from the training data where some
    # machine-written are replaced by attacked essays.
    prompts = []
    for target_text, example_for_detection in tqdm(
        zip(test_data, examples_for_detector),
        total=len(test_data),
        desc=f'Creating prompts for outfox detection considering {attacking_method} attacks...'
    ):
        prompt = make_prompt_for_detection(target_text, example_for_detection)
        prompts.append(prompt)


    # Run the detector for each prompt to detect whether the essay is Human or LM written 
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
    detection_model = args.detection_model
    
    preds_folder = Path(ROOT / "results", "outfox_detection", detection_model)
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
