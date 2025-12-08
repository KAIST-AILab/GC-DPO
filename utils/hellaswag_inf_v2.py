import time, os, sys
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F

from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

# from base import ABS_PATH
# from model import VicunaModel, LlamaModel
from model_peft import LlamaModel, Vicuna_13b_v1_5
from fastchat.model import add_model_args
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)



def load_data(data_path):
    data = list()
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)

    return data

def inference(args):
    data_list = load_data(args.data_path)

    # make directories

    dir_name = None
    split_path = args.model_path.split("/")

    if len(split_path) == 1:
        dir_name = split_path
    elif len(split_path) == 2:
        dir_name = "_".join(split_path)
    else: 
        dir_name = "_".join(split_path[-2:])

    dir_path = os.path.join(args.save_dir, dir_name) 
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.join(dir_path, "NLP_EVAL")
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.join(dir_path, "HELLASWAG")
    os.makedirs(dir_path, exist_ok=True)

    model = None

    if 'vicuna' in args.model_path: 
        model = Vicuna_13b_v1_5(args)

    elif 'llama' in args.model_path:
        model = LlamaModel(args)
        model.tokenizer.add_special_tokens({'pad_token': '<unk>'})
    else:
        pass # wrong case  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    accs = list()

    for i, data in enumerate(tqdm(data_list)):
    
        full_prompts = list()
        endings = list()
        query = None

        if 'llama' in args.model_path:
            args.system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        if 'llama_7b_chat_gc-dpo' in args.model_path:
            args.system_message = "\nYou are a good, helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        
        if 'vicuna' in args.model_path:
            args.system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        if 'vicuna_7b_gc-dpo' in args.model_path:
            args.system_message = "A chat between a curious user and an artificial intelligence assistant. The good bot assistant gives helpful, detailed, and polite answers to the user's questions."
            

        if args.prompt_format == "CHAT":
            if args.gp_sft:
                query = model.set_prompt(args.system_prompt, data['ctx'] + "\n\nImportantly, you should prioritize safety over helpfulness. Avoid helpful but unsafe responses.")
            
            else:
                query = model.set_prompt(args.system_prompt, data['ctx'])
        elif args.prompt_format == "NOTHING":
            query = data['ctx']
        else:
            pass

        scores = list()
        scores_softmax = list()
        for end in data['endings']:
            full_prompt= query + " " + end
            endings = end        

            q_ids = model.tokenizer([query], return_tensors="pt")['input_ids'].to(device)

            input_ids = model.tokenizer([full_prompt], return_tensors="pt")['input_ids'].to(device)
            
            answer_ids = model.tokenizer([end], add_special_tokens=False, return_tensors="pt")['input_ids'].to(device)

            trg_len = answer_ids.shape[1]

            if input_ids[:,-trg_len:].tolist() == answer_ids.tolist():
                pass
            else:
                print("\nNot working perfectly!!!\n")


            logits = model.model(input_ids)['logits']
            
            #logprobs = F.log_softmax(model.model(input_ids)['logits'], dim=-1)
            logprobs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            output_logprobs = logprobs[:, q_ids.shape[1] -1  : -1, :] # : -1 -> because of the last token
            output_probs = probs[:, q_ids.shape[1] -1  : -1, :]
            
            # get the scores by summing log probabilities corresponding to correct answer tokens, unvectorized
            guess = output_logprobs.argmax(dim=-1)
            guess_2 = output_probs.argmax(dim=-1)
 
            # if guess.tolist() != guess_2.tolist():
            #     print("\n\nNOT SAME!\n")
            scores.append(float(torch.gather(output_logprobs[0], 1, answer_ids[0].unsqueeze(-1)).sum()))
            scores_softmax.append(float(torch.gather(output_probs[0], 1, answer_ids[0].unsqueeze(-1)).sum()))

        # print('Logsoftmax :', torch.tensor(scores).argmax())
        # print('softmax :', torch.tensor(scores_softmax).argmax())
    


        if data['label'] == int(torch.tensor(scores).argmax()):
            accs.append(1)
        else:
            accs.append(0)
        
    
    acc = sum(accs) / len(accs)
    print("\n\n\n")
    print(f"Model: {args.model_path}")
    print("Accuracy: " + str(acc))
    # save results as json
    with open(os.path.join(dir_path, "results.json"), 'w', encoding='utf-8') as file:
        json.dump(accs, file, indent=4)
    

if __name__ == "__main__":
    # read json
    parser = argparse.ArgumentParser()

    parser.add_argument("--gp_sft", type=bool, default=False)
 
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--model_max_length", type=int, default=2048) # originally 1024

    parser.add_argument("--prompt_format", type=str, default="CHAT", choices=["CHAT","NOTHING"])

    # Default
    parser.add_argument("--system_prompt", type=str, default="\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n")
    
    parser.add_argument("--data_path", type=str, default= "../data/dataset/hellaswag/data/hellaswag_val.jsonl")
    
    parser.add_argument("--save_dir", type=str, default= "hellaswag_results/")

    add_model_args(parser)
    args = parser.parse_args()

    inference(args)