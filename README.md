# 🔍 SmartSearch: Process Reward-Guided Query Refinement for Search Agents

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Usage](#-usage)
  - [Serving](#serving)
  - [Inference](#inference)
  - [Training](#training)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Overview

Large language model (LLM)-based search agents have proven promising for addressing knowledge-intensive problems by incorporating information retrieval capabilities. Existing works largely focus on optimizing the reasoning paradigms of search agents, yet **the quality of intermediate search queries during reasoning remains overlooked**. As a result, the generated queries often remain inaccurate, leading to unexpected retrieval results and ultimately limiting search agents' overall effectiveness.

**SmartSearch** addresses this challenge through a novel framework built upon two key mechanisms:


1. **Process Rewards**: Provide fine-grained supervision for the quality of each intermediate search query through **Dual-Level Credit Assessment**
2. **Query Refinement**: Promote query generation optimization by selectively refining low-quality search queries and regenerating subsequent search rounds based on these refinements

To enable the search agent to progressively internalize the ability to improve query quality under process reward guidance, we design a **three-stage curriculum learning framework** that guides the agent through a progression from:
- **Imitation** → **Alignment** → **Generalization**




## 📁 Repository Structure

```
SmartSearch/
├── src/                    # Source code for reproducing results
├── scripts/                # Experiment scripts
│   ├── serving/            # Service deployment scripts
│   ├── evaluation/         # Evaluation scripts
│   ├── data_construction/  # Dataset construction scripts
│   └── train/              # Training scripts
├── data/                   # Dataset preprocessing and storage
└── LLaMA-Factory/          # Training framework integration
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Sufficient disk space for datasets and models

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

SmartSearch is trained on the `ASearcher` dataset. The training data can be downloaded from [Hugging Face](XXXX).

To download other test datasets:

```bash
cd data
sh download_dataset.sh
```

To construct the RL dataset:

```bash
cd data
python prepare_dataset.py
```

---

## 💻 Usage

### Serving

#### 1. Sandbox Service

```bash
cd scripts/serving
python sandbox.py --port {port}
```

#### 2. Retriever Service

**Prerequisites:**
- Download [pre-indexed Wikipedia](XXXX)
- Download [Wikipedia corpus and retriever models](XXXX)

**Configuration:**
1. Update `scripts/serving/retriever_config.yaml` with correct paths:
   - Retriever model path
   - Index path
   - Corpus path
   - Available GPU IDs

**Launch:**
```bash
cd scripts/serving
python retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```

#### 3. Model Service

```bash
python3 -m sglang.launch_server \
    --served-model-name {model-name} \
    --model-path {model-path} \
    --tp {tp_num} \
    --dp {dp_num} \
    --context-length 16384 \
    --enable-metrics \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port {port} \
    --trust-remote-code \
    --disable-overlap \
    --disable-radix-cache \
    --mem-fraction-static 0.7
```

### Inference

Ensure all services (sandbox, retriever, and model) are running, then execute:

```bash
cd scripts/evaluation
python run_eval.py \
    --config_path eval_config.yaml \
    --data_dir {data-dir} \
    --dataset_name {dataset-name} \
    --split {split-name} \
    --save_dir {save-dir} \
    --save_note {model-name} \
    --sgl_remote_url {model-url} \
    --remote_retriever_url {retriever-url} \
    --sandbox_url {sandbox-url} \
    --generator_model {model-path}
```

### Training

#### Stage 1: Query Quality Screened Imitation Learning

**Step 1: Trajectory Sampling**

```bash
cd scripts/evaluation
python run_eval.py \
    --config_path eval_config.yaml \
    --data_dir ../../data \
    --dataset_name asearcher \
    --split train \
    --save_dir {save-dir} \
    --save_note {policy-model-name} \
    --sgl_remote_url {policy-model-url} \
    --remote_retriever_url {retriever-url} \
    --sandbox_url {sandbox-url} \
    --generator_model {policy-model-path}
```

**Step 2: Apply Process Rewards**

```bash
cd scripts/data_construction
# Usefulness check by model
python process_reward.py \
    --model_url {process-reward-model-url} \
    --input_file {step1-output-path} \
    --output_file process_reward.json

# Diversity check by rule
python detect_redundancy.py \
    --input_file process_reward.json \
    --output_file process_reward.json
```

**Step 3: Construct SFT Dataset**

```bash
cd scripts/data_construction
python transfer_sft.py \
    --input_file process_reward.json \
    --output_file sft.json
```

**Step 4: SFT Training**

1. Register the dataset in `dataset_info.json`
2. Specify dataset paths in `qwen_full_sft.yaml`

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_full/qwen_full_sft.yaml
```

#### Stage 2: Query Generation Alignment

**Step 1: Trajectory Sampling**

```bash
cd scripts/evaluation
python run_eval.py \
    --config_path eval_config.yaml \
    --data_dir ../../data \
    --dataset_name asearcher \
    --split train \
    --save_dir {save-dir} \
    --save_note {sft-model-name} \
    --sgl_remote_url {sft-model-url} \
    --remote_retriever_url {retriever-url} \
    --sandbox_url {sandbox-url} \
    --generator_model {sft-model-path}
```

**Step 2: Query Refinement**

```bash
cd scripts/data_construction

# Select low-quality queries
python process_reward.py \
    --model_url {process-reward-model-url} \
    --input_file {step1-output-path} \
    --output_file process_reward_1.json

python detect_redundancy.py \
    --input_file process_reward_1.json \
    --output_file process_reward_1.json

# Refine low-quality queries
python query_refinement.py \
    --model_url {process-reward-model-url} \
    --input_file process_reward_1.json \
    --output_file query_refinement.json

# Regenerate subsequent steps
python transfer_generate.py \
    --input_file query_refinement.json \
    --output_file prefix.json

cd ../evaluation
python run_eval.py \
    --config_path eval_config.yaml \
    --data_dir ../../data \
    --dataset_name asearcher \
    --split prefix \
    --save_dir {save-dir} \
    --save_note {sft-model-name} \
    --sgl_remote_url {sft-model-url} \
    --remote_retriever_url {retriever-url} \
    --sandbox_url {sandbox-url} \
    --generator_model {sft-model-path}
```

**Step 3: Construct DPO Dataset**

```bash
python process_reward.py \
    --model_url {process-reward-model-url} \
    --input_file {step2-output-path} \
    --output_file process_reward_2.json

python detect_redundancy.py \
    --input_file process_reward_2.json \
    --output_file process_reward_2.json

python transfer_dpo.py \
    --input_file1 process_reward_1.json \
    --input_file2 process_reward_2.json \
    --output_file dpo.json
```

**Step 4: DPO Training**

1. Register the dataset in `dataset_info.json`
2. Specify dataset paths in `qwen_lora_dpo.yaml`

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/qwen_lora_dpo.yaml
llamafactory-cli export examples/merge_lora/qwen_lora_dpo.yaml
```

#### Stage 3: Query Aware Policy Optimization

```bash
cd scripts/train
bash train.sh \
    --train_batch_size 8 \
    --ppo_mini_batch_size 16 \
    --actor_model_path {dpo-model-path} \
    --search_url {retriever-url} \
    --sandbox_url {sandbox-url} \
    --project_name smart_search \
    --experiment_name smart_search \
    --nnodes 1 \
    --n_gpus_per_node 4 \
    --save_freq 5 \
    --test_freq 5 \
    --total_epochs 2 \
    --wandb_api_key {wandb-api-key} \
    --save_path {save-path} \
    --train_files {train-file-path} \
    --test_files {test-file-path}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We thank the authors of [ReCall](XXXX), [VERL](XXXX), and [FlashRAG](XXXX) for their excellent frameworks that inspired this work.
