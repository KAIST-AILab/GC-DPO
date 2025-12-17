# default system prompt
SYSTEM_PROMPT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
# minimal perturbation
# SYSTEM_PROMPT="A chat between a curious user and an artificial intelligence assistant. The good bot assistant gives helpful, detailed, and polite answers to the user's questions."

MODEL_PATH=(

    "../../checkpoints/llama_7b_chat_gc-dpo_aim_removed_v2_minimal_perturbation_llama_ultrafeedback_200_seed1_shuffle_final_beta_0.03/checkpoint-750"
    "../../checkpoints/vicuna_7b_gc-dpo_aim_removed_v2_minimal_perturbation_llama_ultrafeedback_200_seed1_shuffle_final_beta_0.012/checkpoint-750"
    
)

ABS_PATH=$(dirname "$(realpath "$0")")
DATASET_FILE="/question.jsonl"
# SAVE_DIR="/inf_result"
SAVE_DIR="/../mt_bench_results"
INF_FILE_PATH="/NLP_EVAL/MT-bench/result.jsonl"
JUDGE_FILE_PATH="/NLP_EVAL/MT-bench/gpt-4_single_judge.jsonl"
GPU_ID=0


for MODEL in "${MODEL_PATH[@]}"; do
    echo $MODEL
    MODEL_NAME=$(python -c "
import sys
model_path = sys.argv[1]
split_path = model_path.split('/')
if len(split_path) == 1:
    dir_name = split_path
elif len(split_path) == 2:
    dir_name = '_'.join(split_path)
else: 
    dir_name = '_'.join(split_path[-2:])
print(dir_name)" "$MODEL")
    echo $MODEL_NAME
    CUDA_VISIBLE_DEVICES=$GPU_ID python mt_bench_inf.py \
        --system_prompt "${SYSTEM_PROMPT}" \
        --model_path $MODEL \
        --gp_sft False \
        --dataset_file $ABS_PATH$DATASET_FILE \
        --save_dir $ABS_PATH$SAVE_DIR

    SAVED_PATH="$ABS_PATH$SAVE_DIR/$MODEL_NAME$INF_FILE_PATH"

    CUDA_VISIBLE_DEVICES=$GPU_ID python mt_bench_judgement.py \
        --model_answer_path $SAVED_PATH
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python mt_bench_show.py \
        --input-file "$ABS_PATH$SAVE_DIR/$MODEL_NAME$JUDGE_FILE_PATH"

done
