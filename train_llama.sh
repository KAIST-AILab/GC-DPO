#!bin/bash
HOME_DIR=$(dirname "$(realpath "$0")")
GPU_ID=1
SYSTEM_MESSAGE_PERTURBATION="minimal_perturbation_llama"
FINE_TUNED_MODEL_DIR_BASE="$HOME_DIR/checkpoints/llama_7b_chat_gc-dpo_aim_removed_v2_${SYSTEM_MESSAGE_PERTURBATION}_ultrafeedback_200_seed1_shuffle_final"
FINE_TUNE_DATA_PATH="$HOME_DIR/data/training_data/ultrafeedback_preprocessed_200_seed_1_goal_2_merged.json"
BETA_VALUES=(0.03) 
PORT_NUMBER=(( $GPU_ID + 25000 ))

for BETA in "${BETA_VALUES[@]}"; do
    FINE_TUNED_MODEL_DIR="${FINE_TUNED_MODEL_DIR_BASE}_beta_${BETA}"
    
    ### Fine tune 
    deepspeed --master_port $PORT_NUMBER --include localhost:$GPU_ID $HOME_DIR/train/train_dpo_lora_shuffle_llama.py \
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf  \
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


    #### inference
    DIRECTORIES=(
        "$FINE_TUNED_MODEL_DIR/checkpoint-750"
    )

    BASE_OUTPUT_PATH="$HOME_DIR/inf_result/llama_results"

    for DIRECTORY in "${DIRECTORIES[@]}"; do

        DIR_NAME=$(echo $DIRECTORY | sed 's|.*/checkpoints/||' | sed 's|/|_|g')  # Remove everything before "checkpoint/" and replace slashes

        check_and_run() {
            if [ "$DIRECTORY" == "meta-llama/Llama-2-7b-chat-hf" ] || [ -d "$DIRECTORY" ]; then
                echo "$DIRECTORY exists, running the scripts..."

                for jailbreak_method in none AIM DAN prefix_injection gcg refusal_suppression ; do
                  
                    bot_types=("good bot")

                    for bot_type in "${bot_types[@]}"; do 

                        if [[ "$bot_type" == "good bot" && "${#bot_types[@]}" -eq 1 ]]; then
                            system_message="You are a $bot_type assistant."
                        else
                            system_message="You are a $bot_type assistant."
                        fi
                        
                        suffix="${jailbreak_method}_${bot_type// /}"
                    
                        CUDA_VISIBLE_DEVICES=$GPU_ID python $HOME_DIR/eval/llama_inference_lora.py \
                            --model_path "$DIRECTORY" \
                            --batch_size 1 \
                            --jailbreak_method $jailbreak_method \
                            --repetition_penalty 1.0 \
                            --system_message "$system_message" \
                            --save_path "${BASE_OUTPUT_PATH}/${DIR_NAME}_${suffix}_rep_penalty_1.0.json" \
                            --system_message_perturbation "$SYSTEM_MESSAGE_PERTURBATION" \ 
                            --data_dir "$HOME_DIR/data/advbench_vicuna_7b_dpo_data_aim_processed_eval_list.json"

                        CUDA_VISIBLE_DEVICES=$GPU_ID python labeler_gpt4.py \
                            --file_name "${BASE_OUTPUT_PATH}/${DIR_NAME}_${suffix}_rep_penalty_1.0"

                    done
                done
            else
                echo "$DIRECTORY does not exist, checking again in 120 seconds..."
                sleep 120
                check_and_run
            fi
        }

        check_and_run

    done

done