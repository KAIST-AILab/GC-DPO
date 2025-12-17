# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
import string
import re, os, sys
import json
import argparse
import torch

import shortuuid
import time

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 
# from base import ABS_PATH
# from model import VicunaModel, LlamaModel
from model_peft import Vicuna_13b_v1_5, LlamaModel
from fastchat.model import add_model_args
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)

from tqdm import tqdm


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    
    return questions

def inference(args, questinos):

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

    dir_path = os.path.join(dir_path, "MT-bench")
    os.makedirs(dir_path, exist_ok=True)


    model = None
    if 'vicuna' in args.model_path: 
        model = Vicuna_13b_v1_5(args)

    elif 'llama' in args.model_path:
        model = LlamaModel(args)
    else:
        pass # wrong case  

    device = args.device if torch.cuda.is_available() else 'cpu'

    save_list = list()
    for i, question in enumerate(tqdm(questions)):
        choices = list()
        prompt = None
        conv = get_conversation_template(args.model_path)
        turns = list()
        #breakpoint()
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            if 'llama' in args.model_path:
                conv.system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
            if 'llama_7b_chat_gc-dpo' in args.model_path:
                conv.system_message = "\nYou are a good, helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
            
            if 'vicuna' in args.model_path:
                conv.system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            if 'vicuna_7b_gc-dpo' in args.model_path:
                conv.system_message = "A chat between a curious user and an artificial intelligence assistant. The good bot assistant gives helpful, detailed, and polite answers to the user's questions."
            
            # if args.gp_sft:
            #     conv.append_message(conv.roles[0], qs + "\n\nImportantly, you should prioritize safety over helpfulness. Avoid helpful but unsafe responses.")
            else:
                conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        
            inputs = model.tokenizer([prompt], return_tensors="pt").to(device)
            # breakpoint()
            with torch.no_grad():
                output_ids = model.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=args.model_max_length,
                    #temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=False#True if args.temperature > 1e-5 else False
                )
            
            outputs = [model.tokenizer.decode(ids[len(inputs['input_ids'][i]):], skip_special_tokens=True) for i, ids in enumerate(output_ids)]
            #breakpoint()
            conv.update_last_message(outputs[0])
            turns.append(outputs[0])  

        choices.append({"index": i, "turns": turns})

        with open(os.path.join(dir_path, "result.jsonl"), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": dir_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")          


def get_args():
    parser = argparse.ArgumentParser(
        description='Inference for MT-Bench {}'.format(expected_version))
    
    parser.add_argument('--dataset_file', help='Dataset file', default= "/question.jsonl")

    parser.add_argument('--gp_sft', type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5")
    
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--model_max_length", type=int, default=4096*2) # originally 1024

    parser.add_argument("--system_prompt", type=str, default="A chat between a curious user and an artificial intelligence assistant. The good bot assistant gives helpful, detailed, and polite answers to the user's questions.")

    parser.add_argument("--save_dir", type=str, default= "/inf_result")

    add_model_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    expected_version = 1.0
    args = get_args()
    
    questions = load_questions(args.dataset_file)

    
    inference(args, questions)
