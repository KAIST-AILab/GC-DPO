import os
import argparse
import json
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from concurrent.futures import ThreadPoolExecutor, as_completed


device = "cuda"
model_name = "gpt2-xl"

def initialize_model_and_tokenizer():
    # Load the model and tokenizer for each thread separately
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    return model, tokenizer

def calculate_perplexity(model, tokenizer, text, filename):
    
    if len(text) <= 2:
        sos_token = "[SOS]"
        text = sos_token + " " + text
        print(f"Added SOS token for short text: {text}")

    encodings = tokenizer(text, return_tensors='pt').to(device)
    max_length = model.config.n_positions
    stride = 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        # Filter out nan values
        if torch.isnan(neg_log_likelihood):
            continue
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:  # If nlls is empty, return nan to avoid issues
        print(f"No valid likelihoods for filename: {filename}")
        print(f"No valid likelihoods for text: {text}")
        breakpoint()
        return torch.tensor(float('nan'))

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

def process_json_files(directory):
    model, tokenizer = initialize_model_and_tokenizer()  # Each thread gets its own model and tokenizer
    total_perplexity_sum = 0
    total_field_count = 0

    text_fields = ['llama_output', 'vicuna_output', 'output']  # Fields to analyze

    # Get all the JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith("labeled_new.json")]
    # json_files = [f for f in os.listdir(directory) if f.endswith("_goodbot.json")]
    json_files = [f for f in json_files if "rot_" not in f]
    json_files = [f for f in json_files if "base64_" not in f]
    json_files = [f for f in json_files if "badbot_" not in f]
    text_fields = ['llama_output', 'output', 'vicuna_output']  # Specific fields to analyze

    with open('../test_id_list.json', 'r') as file:
        test_id_list = json.load(file)

    for filename in tqdm(json_files, desc="Processing Files"):
        file_path = os.path.join(directory, filename)

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data_list = json.load(file)  # Assume data is a list of dictionaries

            # Iterate through each dictionary in the list
            for data in tqdm(data_list, desc=f"Processing Data in {filename}", leave=False):
                # Process each specified text field separately
                # if 'id' in data and data['id'] in test_id_list:
                if True:
                    for field in text_fields:
                        if field in data and data[field].strip():
                            text = data[field].strip()
                            if len(text.split()) < 2:
                                continue

                            # Calculate perplexity for the text using thread-local model
                            perplexity = calculate_perplexity(model, tokenizer, text, filename)

                            # Accumulate the sum of perplexities
                            total_perplexity_sum += perplexity
                            total_field_count += 1

    # Calculate the average perplexity
    average_perplexity = total_perplexity_sum / total_field_count if total_field_count > 0 else 0
    return average_perplexity, total_field_count

def process_directory(directory):
    average_perplexity, total_field_count = process_json_files(directory)
    print(f"Average perplexity in {directory}: {average_perplexity}")
    print(f"Total field count in {directory}: {total_field_count}")

    # Save the results to a file
    with open(f"/home/jbmaeng/HER_2H/utils/perplexity_output.txt", "a") as f:
        f.write(f"Perplexity in {directory}\n")
        f.write(f"measured with model: {model_name}\n")
        f.write(f"Average perplexity: {average_perplexity}\n")
        f.write(f"Total field count: {total_field_count}\n")
        f.write("-" * 50)
        f.write("\n")

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)

    current_directory = os.path.dirname(current_file_path)
    
    directory_paths = [os.path.join(current_directory, "../inf_result/vicuna_results"), os.path.join(current_directory, "../inf_result/llama_results")]
    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=8) as executor:
        
        futures = {executor.submit(process_directory, directory): directory for directory in directory_paths}

        # Progress tracking for threads
        for future in as_completed(futures):
            directory = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Directory {directory} generated an exception: {exc}")
