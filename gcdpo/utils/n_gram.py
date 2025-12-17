import nltk
from nltk.util import ngrams
from collections import Counter
import os, sys 

import json

# Ensure necessary resources are available
# nltk.download('punkt')
nltk.download('punkt_tab')

def generate_ngrams(text, n):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text.lower())
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Return n-grams and the total number of n-grams generated
    return n_grams, len(n_grams)

def count_ngrams(ngrams_list):
    # Count the frequency of each n-gram
    ngram_counts = Counter(ngrams_list)
    
    # Calculate the number of unique n-grams
    unique_ngrams = len(ngram_counts)
    
    # Calculate the total number of n-grams
    total_ngrams = sum(ngram_counts.values())
    
    # Calculate the ratio of unique n-grams to total n-grams
    unique_to_total_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    
    return unique_ngrams, total_ngrams, unique_to_total_ratio

def process_json_files(directory, n):
    total_ratio_sum = 0
    total_field_count = 0
    
    text_fields = ['llama_output', 'output', 'vicuna_output']  # Specific fields to analyze
    with open('../test_id_list.json', 'r') as file:
        test_id_list = json.load(file)

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data_list = json.load(file)  # Assume data is a list of dictionaries
                
                # Iterate through each dictionary in the list
                for data in data_list:
                    # breakpoint()
                    # if 'id' in data and data['id'] in test_id_list:
                    if True:
                        # Process each specified text field separately
                        for field in text_fields:
                            if field in data:
                                text = data[field]
                                # id_list.append(data['id'])
                                
                                # Generate n-grams and get the total number of n-grams
                                ngrams_list, total_ngrams = generate_ngrams(text, n)
                                
                                # Calculate the unique n-grams and ratio
                                _, _, unique_to_total_ratio = count_ngrams(ngrams_list)
                                
                                # Accumulate the sum of the unique-to-total ratios
                                total_ratio_sum += unique_to_total_ratio
                                total_field_count += 1

    average_ratio = total_ratio_sum / total_field_count if total_field_count > 0 else 0

    return average_ratio, total_field_count


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)

    current_directory = os.path.dirname(current_file_path)
    
    directory_paths = [os.path.join(current_directory, "../inf_result/vicuna_results"), os.path.join(current_directory, "../inf_result/llama_results")]
    # directory_paths = ["inf_result/llama_results", "inf_result/llama_results"]
    for directory_path in directory_paths:
        print(f"\n\n{directory_path} result:")
        for n in [2,3]:
            # Calculate the average unique n-gram ratio for all specified fields in all JSON files
            average_ratio, processed_fields = process_json_files(directory_path, n)

            # Display the result
            print(f"Processed {processed_fields} fields across all files")
            print(f"Average unique-to-total {n}-gram ratio: {average_ratio}")


