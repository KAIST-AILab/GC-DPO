# Goal-Conditioned DPO (GC-DPO)

Official experimental repository for the NAACL 2025 paper:
> **Goal-Conditioned DPO: Prioritizing Safety in Misaligned Instructions**  
> Joo Bon Maeng*, Seongmin Lee*, Seokin Seo, Kee-Eung Kim  
> NAACL 2025 (Long Papers)

This repository provides code to reproduce the experiments in the paper, including
goal-conditioned data construction, GC-DPO training, and evaluation against jailbreak attacks.


### Overview
Large Language Models (LLMs) are vulnerable to **jailbreak attacks** that exploit
misalignment between system prompts (safety goals) and user prompts.
GC-DPO addresses this problem by explicitly learning **instruction hierarchy**
during training.

Key ideas:
- Decompose input into **system prompt (goal)** and **user prompt**
- Condition preference optimization on the system-level goal
- Learn to prioritize safety goals over misaligned user instructions

GC-DPO significantly reduces jailbreak attack success rates while preserving
general task performance.

### Method Summary
GC-DPO extends Direct Preference Optimization (DPO) by conditioning preference
ordering on a goal variable defined in the system prompt.

### Goals
- `gGOOD`: prioritize safety and ethical constraints
- `gBAD`: prioritize user instruction (used only during training)

For the same user prompt `u`, the preference between safe and harmful responses
is reversed depending on the goal, enabling the model to learn prompt hierarchy.

---


## Env

### Environment Setting
* We use python version 3.9.19 for all our experiments
conda env create -n gcdpo -f environment.yaml
conda activate gcdpo
cd gcdpo

### API
* As the models we used are OpenAI models, you should configure your own OPENAI API KEY.
* You must set your own OPENAI API KEY in the base.py and export it as your environmental variable using 
```
export OPENAI_API_KEY=[YOUR API KEY]
```

**Disclaimer: Due to the nature of this research the following sections will contain potentially harmful contents**
## Data Generation

* Inference the data with None and AIM on advbench dataset  
* Evaluate the generated data with the Judge model

   ( The results are already provided in the /data directory  

    data/advbench_vicuna_7b_dpo_data_aim_processed.json )
```
bash data_generation.sh
```

## GC-DPO Training & Evaluation
* The training phase and the evaluation against jailbreak attacks are combined into one script per model for easier execution, which are:
```
bash train_vicuna.sh
bash train_llama.sh
```

## General Task Performance
* One we have obtained the trained model weights we can evaluate the various performance metrics/benchmarks.
* First, we need to change the current directory to utils with the following command
```
cd utils
```

* Then, we measure general task performance along with some benchmarks in a single bash file (n-gram, perplexity, TriviaQA, HellaSwag)

```
bash general_task_performance.sh
```

## MT-Bench 
* In order to measure the MT-Bench score, we need to change the current directory to mt_bench
```
cd mt_bench
```
* Then, run the corresponding bash file to execute the evaluation
```
bash mt_bench_eval.sh
```


* You can check the results in the [TASK_NAME]_results directories

**Note that all model weights and datasets used for training and evaluation are included in the provided files**


## Citation

```bibtex
@inproceedings{maeng-etal-2025-goal,
  title={Goal-Conditioned {DPO}: Prioritizing Safety in Misaligned Instructions},
  author={Maeng, Joo Bon and Lee, Seongmin and Seo, Seokin and Kim, Kee-Eung},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year={2025}
}
