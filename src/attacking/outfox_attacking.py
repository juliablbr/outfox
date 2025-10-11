import random
import sys
from tqdm import tqdm
from src.utils import ROOT
from src.utils.utils import (
    OutfoxDetectionLLM,
    TaskTriplet,
    combine_machine_text_label,
    get_similar_triplets,
    identify_attack_labels,
    make_prompts_for_attacks,
    parse_outfox_args,
    generation_by_vllm,
    load_pkl,
    save_pkl,
)


def make_attacked_test(
    llm,
    test_humans,
    test_problem_statements,
    test_ps_to_near_ps_human_lm_pairs_from_train,
    train_ps_to_near_ps_human_lm_pairs_from_train,
):
    # prompts = []
    # max_tokens = 0
    # for test_human, test_ps in tqdm(zip(test_humans, test_problem_statements), total=len(test_humans)):
    #    example_for_detection = test_ps_to_near_ps_human_lm_pairs_from_train[test_ps]
    #    lm_and_attack_labels = []
    #    for train_ps, _, train_lm in tqdm(example_for_detection):
    #        lm_and_attack_label = identify_attack_label(llm, train_ps, train_lm, train_ps_to_near_ps_human_lm_pairs_from_train)
    #        lm_and_attack_labels.append(lm_and_attack_label)
    #    assert len(lm_and_attack_labels) == 10, 'invalid size of lm_and_attack_labels'
    #    random.shuffle(lm_and_attack_labels)
    #    prompt, human_essay_tokens = make_prompt_for_attack(lm_and_attack_labels, test_ps, test_human)
    #    prompts.append(prompt)
    #    max_tokens = max([min([4000, human_essay_tokens]), max_tokens])

    task_triplets = [
        [
            TaskTriplet(test_ps, test_human, "")
            for test_human, test_ps in zip(test_humans, test_problem_statements)
        ]
    ]

    similar_examples_to_attacked: list[list[list[TaskTriplet]]] = get_similar_triplets(
        task_triplets,
        test_ps_to_near_ps_human_lm_pairs_from_train,
        n_similar_triplets=10,
    )

    attack_labels = identify_attack_labels(
        llm, similar_examples_to_attacked, train_ps_to_near_ps_human_lm_pairs_from_train
    )
    attacker_examples = combine_machine_text_label(
        similar_examples_to_attacked, attack_labels
    )

    prompts, human_essay_tokens = make_prompts_for_attacks(
        task_triplets, attacker_examples
    )
    # This should have temperature 1.3 and top_p=1 if we want to mimic the paper, that was the original code. So probably add those values
    # As command line arguments!
    # temperature=1.3, top_p=1
    attacked_essays = generation_by_vllm(
        prompts,
        llm,
        {
            "temperature": 1.3,
            "top_p": 1.0,
            "max_tokens": max(human_essay_tokens),
        },
    )
    return attacked_essays


def main():
    args = parse_outfox_args().parse_args()

    random.seed(args.seed)

    if args.dataset == "outfox":
        test_human = "data/common/test/test_humans.pkl"
        test_problem_statements = "data/common/test/test_problem_statements.pkl"
        test_train_problem_statement_map = (
            "data/chatgpt/util/test_ps_to_near_ps_human_lm_pairs_from_train.pkl"
        )
        train_train_problem_statement_map = (
            "data/chatgpt/util/train_ps_to_near_ps_human_lm_pairs_from_train.pkl"
        )
        target_file = "data/chatgpt/test/test_outfox_attacks.pkl"
    else:
        dataset_name = args.dataset
        dataset_folder = ROOT / f"{dataset_name}_pickles"
        test_human = (
            dataset_folder
            / f"{'combined' if dataset_name == 'm4' else dataset_name}_humans_test.pkl"
        )
        test_problem_statements = (
            dataset_folder
            / f"{'combined' if dataset_name == 'm4' else dataset_name}_problem_statements_test.pkl"
        )
        test_train_problem_statement_map = (
            dataset_folder / "test_ps_to_near_ps_human_lm_pairs_from_train.pkl"
        )
        train_train_problem_statement_map = (
            dataset_folder / "train_ps_to_near_ps_human_lm_pairs_from_train.pkl"
        )
        target_file = dataset_folder / "test_outfox_attacks.pkl"

    test_humans = load_pkl(test_human)
    test_problem_statements = load_pkl(test_problem_statements)

    # Loading top-k (problem statement, human-written essay, LLM-generated essay) sets retrieved in advance based on the problem statement, using tf-idf.
    # Whatever LMs (FLAN, GPT-3.5) to be detected, our OUTFOX detector consider the essays by ChatGPT.
    test_ps_to_near_ps_human_lm_pairs_from_train = load_pkl(
        test_train_problem_statement_map
    )
    train_ps_to_near_ps_human_lm_pairs_from_train = load_pkl(
        train_train_problem_statement_map
    )

    llm = OutfoxDetectionLLM(args)
    attacked_test = make_attacked_test(
        llm,
        test_humans,
        test_problem_statements,
        test_ps_to_near_ps_human_lm_pairs_from_train,
        train_ps_to_near_ps_human_lm_pairs_from_train,
    )
    save_pkl(attacked_test, target_file)
    print("Complete!")


if __name__ == "__main__":
    main()
