from dataclasses import dataclass
from numpy import dot
from numpy.linalg import norm
import pickle
import json
import random
import torch
import tiktoken
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ollama
import asyncio
from ollama import AsyncClient
from tqdm import tqdm
from pydantic import BaseModel
import csv
from pathlib import Path

print('Loading tiktoken tokenizer...')
gpt_turbo_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


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


@dataclass
class OllamaHostConfig:
    model_name: str
    parallel_requests_per_host: int
    number_of_hosts: int
    host_ip: str
    host_start_port: int


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
        truncated_encodes = gpt_turbo_encoding.encode(text)[:truncated_length]
        return gpt_turbo_encoding.decode(truncated_encodes)
    elif truncate_type == 'word':
        return ' '.join(text.split(' ')[:truncated_length])


def string2token_nums(string):
    num_tokens = 0
    num_tokens = len(gpt_turbo_encoding.encode(string))
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


async def generation_by_ollama(ollama_host_config: OllamaHostConfig, prompts, **kwargs):
    progress_bar = tqdm(total=len(prompts), desc="Generating the ollama responses...", position=0)
    # We append a bunch of None values to preserve order
    results = [None] * len(prompts)
    # Create async queue with all Messages (prompts)
    task_queue = asyncio.Queue()
    for idx, message in enumerate(prompts):
        task_queue.put_nowait((idx, message))

    tasks = []
    host_start_port = ollama_host_config.host_start_port
    n_hosts = ollama_host_config.number_of_hosts
    parallel_requests = ollama_host_config.parallel_requests_per_host

    worker_count = n_hosts * parallel_requests
    for i in range(worker_count):
        host_port = host_start_port + (i // parallel_requests)
        host = f"{ollama_host_config.host_ip}:{host_port}"
        print(f"Worker {i} on host {host}")

        task = asyncio.create_task(worker(ollama_host_config.model_name, task_queue, results, host, progress_bar, **kwargs))
        tasks.append(task)
    
    await task_queue.join()
    
    for task in tasks:
        task.cancel()
    
    # Await the completion of all worker tasks
    await asyncio.gather(*tasks)
    progress_bar.close()

    return results

async def worker(model, task_queue, results, host, progress_bar, **kwargs):
    try:
        while not task_queue.empty():
            idx, message = await task_queue.get()
            answer = await start_ollama_chat(message, model, host, **kwargs)
            if answer is None:
                results[idx] = None
            else:
                results[idx] = answer.detection
            progress_bar.update(1)
            task_queue.task_done()
    except asyncio.CancelledError:
        print("asyncio.CancelledError")
        return

async def start_ollama_chat(message, model, host, **kwargs):
    response = await AsyncClient(host=host).chat(
        model=model,
        messages=[{"role": "user", "content": f"{message}"}],
        options=kwargs if kwargs else None,
        format=LLMAnswer.model_json_schema(),
    )
    try:
        answer = LLMAnswer.model_validate_json(response.message.content)
    except Exception as e:
        print(f"[error] {e}")
        return None
    return answer

def make_prompt_for_detection(target_text, pairs):
    instruction = 'Please classify whether the text is generated by a Human or a Language Model(LM).\n'
    assert len(pairs) == 5, 'Incorrect example nums'
    prompt = instruction
    ### Truncating text depending on the maximum length.
    truncated_length = (4000 - (string2token_nums(instruction) + (string2token_nums(f'Text: Answer: Human\n') + string2token_nums(f'Text: Answer: LM\n')) * 5 + string2token_nums(f'Text: {target_text} Answer: '))) // 10
    all_data = []
    for pair in pairs:
        human, lm = pair[-2], pair[-1]
        human, lm = truncate_text(human, truncated_length, 'token'), truncate_text(lm, truncated_length, 'token')
        all_data.append((human, 'Human'))
        all_data.append((lm, 'LM'))
    random.shuffle(all_data)
    for essay, label in all_data:
        prompt += f'Text: {essay} Answer: {label}\n'
    prompt += f'Text: {target_text} Answer: '
    return prompt


def make_prompt_for_attack(lm_and_attack_labels, problem_statement, human_essay):
    human_essay_words = len(human_essay.split(' '))
    human_essay_tokens = string2token_nums(human_essay)
    instruction_1 = f'Here are the results of detecting whether each essay from each problem statement is generated by a Human or a Language Model(LM).\n'
    instruction_2 = f'\nGiven the following problem statement, please write an essay detected as Human in {human_essay_words} words with a clear opinion.\n\nProblem statement:\n{problem_statement}\nAnswer:\nHuman\nEssay:\n'
    prompt = instruction_1

    all_ps_length = sum([string2token_nums(ps) for ps, _, _ in lm_and_attack_labels])
    truncated_length = (4000 - (string2token_nums(instruction_1 + instruction_2) + (string2token_nums(f'Problem statement:\n\nAnswer:\nHuman\nEssay:\n\n') + string2token_nums(f'Problem statement:\n\nAnswer:\nLM\nEssay:\n\n')) * 5 + all_ps_length + human_essay_tokens + 50)) // 10
    assert len(lm_and_attack_labels) == 10, 'Invalid test data size'

    for ps, lm_essay, attack_label in lm_and_attack_labels:
        lm_essay = truncate_text(lm_essay, truncated_length, 'token')
        prompt += f"Problem statement:\n{ps}\nAnswer:\n{'Human' if attack_label == 'Good' else 'LM'}\nEssay:\n{lm_essay}\n"
    prompt += instruction_2
    return (prompt, human_essay_tokens)


def identify_attack_label(ps, lm, train_ps_to_near_ps_human_lm_pairs_from_train):
    # Attack label includes "Good" or "Bad".
    # 'Good' example for attacker is an LLM-generated essay which our detector misclassify, retrieved from the training set.
    # 'Bad' example for attacker is an LLM-generated essay which our detector classify correctly, retrieved from the training set
    ps_human_lm_train = train_ps_to_near_ps_human_lm_pairs_from_train[ps]
    human_lm_train = [(human, lm) for ps, human, lm in ps_human_lm_train]
    prompt = make_prompt_for_detection(lm, human_lm_train[:5])
    res = generation_by_ollama(prompt, temperature=0, top_p=0)
    pred = process_reply_from_chatgpt(res)
    return (ps, lm, 'Good' if pred == '0' else 'Bad')


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

