## GC-DPO

## Env

### Python Installation
* We use python version 3.9.19 for all our experiments
```
pip install -r requirements.txt
```

## API
* As the models we used are OpenAI models, you should configure your own OPENAI API KEY.
* You must set your own OPENAI API KEY in the base.py and export it as your environmental variable using 
```
export OPENAI_API_KEY=[YOUR API KEY]
```

**Disclaimer: Due to the nature of this research the following sections will contain potentially harmful contents**
## Data geneartion

* Inference the data with None and AIM on advbench dataset  
* Evaluate the generated data with the Judge model

   ( The results are already provided in the /data directory  

    Your_repository/data/advbench_vicuna_7b_dpo_data_aim_processed.json )
```
bash data_generation.sh
```

## GC-DPO Training & Evaluation
* The training phase and the evaluation against jailbreak attacks are combined into one script per model for easier execution, which are:
```
bash train_vicuna.sh
bash train_llama.sh
```

## General task performance
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
@inproceedings{maeng2025gcdpo,
  title={Goal-Conditioned DPO: Prioritizing Safety in Misaligned Instructions},
  author={Maeng, Joo Bon and Lee, Seongmin and Seo, Seokin and Kim, Kee-Eung},
  booktitle={Proceedings of NAACL 2025},
  year={2025}
}
