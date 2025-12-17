import os
import sys
import argparse
from datasets import load_dataset
import json

def reformat_data(dataset, 
                  goals=["You are a good bot assistant.", "You are a bad bot assistant."], 
                  split_size=10000,
                  seed=42):

    data2save = []
    dataset_shuffled = dataset["train"].shuffle(seed=seed)
    dataset = dataset_shuffled.select(range(split_size))
    for row in dataset:
        for goal in goals:
            data2save.append({
                "prompt": row["prompt"],
                "goal": goal,
                "chosen": row["chosen"][1]['content'],
                "rejected": row["rejected"][1]['content']
            })
    return data2save

def merge_two_jsons(json1, json2):
    return json1 + json2
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--goals", type=list, default=["You are a good bot assistant.",
                                                       "You are a bad bot assistant."])
    
    parser.add_argument("--split_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

    reformatted_data = reformat_data(ds, 
                                     goals=args.goals, 
                                     split_size=args.split_size, 
                                     seed=args.seed)
    
    with open(os.path.join(args.output_dir, f"ultrafeedback_preprocessed_{args.split_size}_seed_{args.seed}_goal_{len(args.goals)}.json"), "w") as f:
        json.dump(reformatted_data, f, indent=4)
    
    print("Data saved to", os.path.join(args.output_dir, f"ultrafeedback_preprocessed_{args.split_size}_seed_{args.seed}_goal_{len(args.goals)}.json"))

    # merge with the original dataset and save to another file

    with open(os.path.join(args.data_dir, "advbench_vicuna_7b_dpo_data_aim_processed.json"), "r") as f:
        aim_data = json.load(f)
    
    reformatted_data = merge_two_jsons(reformatted_data, aim_data)

    with open(os.path.join(args.data_dir, f"training_data/ultrafeedback_preprocessed_{args.split_size}_seed_{args.seed}_goal_{len(args.goals)}_merged.json"), "w") as f:
        json.dump(reformatted_data, f, indent=4)
