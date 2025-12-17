#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

python $SCRIPT_DIR/utils/preprocess_ultrafeedback_dataset.py \
    --output_dir $SCRIPT_DIR/data/dataset/ultrafeedback-binarized-preferences-cleaned/data \
    --data_dir $SCRIPT_DIR/data \
    --seed 1 



