#!/bin/bash

HOME_DIR=$(dirname "$(realpath "$0")")

DIRS=(
    "../checkpoints/vicuna_7b_gc-dpo_aim_removed_v2_minimal_perturbation_ultrafeedback_200_seed1_shuffle_beta_0.012/checkpoint-750"
    "../checkpoints/llama_7b_chat_gc-dpo_aim_removed_v2_minimal_perturbation_llama_ultrafeedback_200_seed1_shuffle_final_beta_0.03/checkpoint-750"
)
    

##### n-gram
python n_gram.py

# ##### perplexity
python perplexity_batch.py

for DIR in "${DIRS[@]}"; do
    FULL_PATH="$HOME_DIR/$DIR"

    # # HellaSwag task
    python hellaswag_inf_v2.py --model_path "$FULL_PATH"

    # TriviaQA task
    python triviaqa_inf.py --model_path "$FULL_PATH"
done
