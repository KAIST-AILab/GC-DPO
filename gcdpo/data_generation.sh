#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

python run_batch_inference.py --jailbreak=None
python run_batch_inference.py --jailbreak=AIM

python labeler_gpt4.py --file_path="$SCRIPT_DIR/inf_result/lmsys_vicuna-7b-v1.5/None/result.json"
python labeler_gpt4.py --file_path="$SCRIPT_DIR/inf_result/lmsys_vicuna-7b-v1.5/AIM/result.json"

python save_dpo_data.py \
    --inf_data_dir="$SCRIPT_DIR/inf_result/lmsys_vicuna-7b-v1.5/None/result_labeled.json" \
    --inf_aim_data_dir="$SCRIPT_DIR/inf_result/lmsys_vicuna-7b-v1.5/AIM/result_labeled.json"

python $SCRIPT_DIR/utils/preprocess_ultrafeedback_dataset.py \
    --output_dir $SCRIPT_DIR/data/dataset/ultrafeedback-binarized-preferences-cleaned/data \
    --data_dir $SCRIPT_DIR/data \
    --seed 1 
