# OUTFOX Fork🦊🍴
![](https://img.shields.io/badge/Made_with-python-blue.svg)
![](https://img.shields.io/badge/License-Apache--2.0-green.svg)

This repository contains an enhanced implementation of [OUTFOX](https://github.com/ryuryukke/OUTFOX), a framework for improving the robustness of machine-generated text detectors. Our work extends the original OUTFOX paper by Koike et al. (2024) with out-of-domain evaluations, vLLM integration, and performance optimizations.

## Installation
- `python==3.9.2`

1. Clone the repository:
```bash
$ git clone https://github.com/juliablbr/outfox.git
$ cd outfox
```
2. Install dependencies
```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Usage

### Basic Detection (Without Attack Consideration)

For standard detection without attack consideration:
```bash
$ python -m src.detection.outfox_detection_without_considering_attack \
  --attacking_method "$ATTACK_METHOD" \
  --model_name "$MODEL" \
  --detection_model "$DETECTION_MODEL" \
  --max_model_len 8192 \
  --seed "$SEED"
```

### Advanced Detection (With Attack Consideration)

For detection with attack consideration:
```bash
$ python -m src.detection.outfox_detection_with_considering_attack_new \
  --attacking_method "$ATTACK_METHOD" \
  --model_name "$MODEL" \
  --detection_model "$DETECTION_MODEL" \
  --max_model_len 8192 \
  --seed "$SEED"
```

### Cross-Domain Evaluation

To evaluate on new datasets without attack consideration:
```bash
$ python -m src.detection.outfox_detection_without_considering_attack \
  --dataset "$DATASET" \
  --detection_model "$DETECTION_MODEL" \
  --max_model_len 8192 \
  --seed "$SEED"
```

To evaluate on new datasets with attack consideration:
```bash
$ python -m src.detection.outfox_detection_with_considering_attack_new \
  --dataset "$DATASET" \
  --detection_model "$DETECTION_MODEL" \
  --max_model_len 8192 \
  --seed "$SEED"
```

Parameters
- `--attacking_method`: Attack method to use
- `--model_name`: OUTFOX dataset to use
- `--detection_model`: Detection model to use
- `--dataset`: Dataset for out-of-domain evaluation
- `--max_model_len`: Maximum model context length (default: 8192)
- `--seed`: Random seed for reproducibility

## Enhancements

Our implementation includes several key improvements over the original OUTFOX:

1. Out-of-Domain Evaluation: Extended robustness assessment beyond the essay domain
2. vLLM Integration: Replaced OpenAI API with vLLM for running models locally
3. Batch Processing: Modified code flow to support batching for improved runtime performance
