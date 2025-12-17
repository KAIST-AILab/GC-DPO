#!/bin/bash
# Script for training and evaluating Vicuna-7B model with DPO (Direct Preference Optimization)
# This script performs fine-tuning with LoRA and then evaluates the model on various jailbreak methods

# Get the directory where this script is located
HOME_DIR=$(dirname "$(realpath "$0")")

# Configuration variables
GPU_ID=0  # GPU device ID to use for training and inference
SYSTEM_MESSAGE_PERTURBATION="minimal_perturbation"  # Type of system message perturbation to apply
FINE_TUNED_MODEL_DIR_BASE="$HOME_DIR/checkpoints/vicuna_7b_gc-dpo_aim_removed_v2_${SYSTEM_MESSAGE_PERTURBATION}_ultrafeedback_200_seed1_shuffle"
FINE_TUNE_DATA_PATH="$HOME_DIR/data/training_data/ultrafeedback_preprocessed_200_seed_1_goal_2_merged.json"
BETA_VALUES=(0.012)  # DPO beta values to experiment with (controls the strength of preference optimization)
PORT_NUMBER=$(( $GPU_ID + 25000 ))  # DeepSpeed master port number (calculated from GPU ID)

# Loop through different beta values for DPO training
for BETA in "${BETA_VALUES[@]}"; do
    # Create a unique output directory for each beta value
    FINE_TUNED_MODEL_DIR="${FINE_TUNED_MODEL_DIR_BASE}_beta_${BETA}"
    
    ### Fine-tuning stage: Train the model using DPO with LoRA
    # Use DeepSpeed for distributed training with mixed precision
    deepspeed --master_port $PORT_NUMBER --include localhost:$GPU_ID $HOME_DIR/train/train_dpo_lora_shuffle_vicuna.py \
        --model_name_or_path lmsys/vicuna-7b-v1.5  \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --data_path $FINE_TUNE_DATA_PATH \
        --output_dir $FINE_TUNED_MODEL_DIR \
        --num_train_epochs 5 \
        --fp16 True \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 150 \
        --save_total_limit 100 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_strategy "steps" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --q_lora False \
        --deepspeed $HOME_DIR/utils/ds_config_s3.json \
        --gradient_checkpointing True \
        --flash_attn True \
        --beta $BETA 


    #### Inference stage: Evaluate the fine-tuned model on various jailbreak methods
    # List of checkpoint directories to evaluate (typically the best checkpoint from training)
    DIRECTORIES=(
        "$FINE_TUNED_MODEL_DIR/checkpoint-750"
    )

    # Base path for saving inference results
    BASE_OUTPUT_PATH="$HOME_DIR/inf_result/vicuna_results"

    # Loop through each checkpoint directory
    for DIRECTORY in "${DIRECTORIES[@]}"; do
        # Extract directory name for use in output filenames
        # Remove everything before "checkpoints/" and replace slashes with underscores
        DIR_NAME=$(echo $DIRECTORY | sed 's|.*/checkpoints/||' | sed 's|/|_|g')

        # Function to check if checkpoint exists and run evaluation
        check_and_run() {
            # Check if the directory exists (or if it's the base model path)
            if [ "$DIRECTORY" == "lmsys/vicuna-7b-v1.5" ] || [ -d "$DIRECTORY" ]; then
                echo "$DIRECTORY exists, running the scripts..."

                # Test the model against various jailbreak attack methods
                for jailbreak_method in none AIM DAN prefix_injection gcg refusal_suppression ; do
                  
                    # Bot types to test (e.g., "good bot", "bad bot", etc.)
                    bot_types=("good bot")

                    # Loop through different bot type configurations
                    for bot_type in "${bot_types[@]}"; do 
                        # Construct system message based on bot type
                        if [[ "$bot_type" == "good bot" && "${#bot_types[@]}" -eq 1 ]]; then
                            system_message="You are a $bot_type assistant."
                        else
                            system_message="You are a $bot_type assistant."
                        fi
                        
                        # Create suffix for output filename
                        suffix="${jailbreak_method}_${bot_type// /}"
                    
                        # Run inference with the fine-tuned model
                        # Test model's resistance to jailbreak attacks
                        CUDA_VISIBLE_DEVICES=$GPU_ID python $HOME_DIR/eval/vicuna_inference_lora.py \
                            --model_path "$DIRECTORY" \
                            --batch_size 1 \
                            --jailbreak_method $jailbreak_method \
                            --system_message "$system_message" \
                            --save_path "${BASE_OUTPUT_PATH}/${DIR_NAME}_${suffix}.json" \
                            --system_message_perturbation "$SYSTEM_MESSAGE_PERTURBATION" \
                            --data_dir "$HOME_DIR/data/advbench_vicuna_7b_dpo_data_aim_processed_eval_list.json"

                        # Use GPT-4 to label/evaluate the inference results
                        CUDA_VISIBLE_DEVICES=$GPU_ID python $HOME_DIR/utils/labeler_gpt4.py \
                            --file_name "${BASE_OUTPUT_PATH}/${DIR_NAME}_${suffix}"

                    done
                done
            else
                # If checkpoint doesn't exist yet, wait and retry (useful if training is still running)
                echo "$DIRECTORY does not exist, checking again in 120 seconds..."
                sleep 120
                check_and_run
            fi
        }

        # Execute the check and run function
        check_and_run

    done  # End of checkpoint directory loop

done  # End of beta values loop