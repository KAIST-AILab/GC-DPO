# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re, os, sys
import json
import argparse
import torch

from os.path import dirname, join, abspath

# sys.path.insert(0, abspath(join(dirname(__file__)))) 


from model_peft import Vicuna_13b_v1_5, LlamaModel

from fastchat.model import add_model_args
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
#import utils.dataset_utils
#import utils.utils
import triviaqa_utils
from tqdm import tqdm

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def is_exact_match(answer_object, prediction):
    ground_truths = get_ground_truths(answer_object)
    for ground_truth in ground_truths:
        if exact_match_score(prediction, ground_truth):
            return True
    return False


def has_exact_match(ground_truths, candidates):
    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False


def get_ground_truths(answer):
    return answer['NormalizedAliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])]


def get_oracle_score(ground_truth, predicted_answers, qid_list=None, mute=False):
    exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = normalize_answer(predicted_answers[qid])
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = has_exact_match(ground_truths, prediction)
        exact_match += int(em_for_this_question)

    exact_match = 100.0 * exact_match / len(qid_list)

    return {'oracle_exact_match': exact_match, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}


def evaluate_triviaqa(ground_truth, predicted_answers, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Missed question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predicted_answers[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}

'''Answer these question:
        Question: In Scotland a bothy/bothie is a?
        Answer: House
        Question: Who sang who wants to be a millionarie in high society?
        Answer: Frank Sinatra'''
def llama_triviaqa_zero_prompt(question):
    prompt = f'''Answer the question.
Qustion: {question}
Answer:'''
    return prompt

def inference(args, dataset):

    dir_name = None
    split_path = args.model_path.split("/")

    if len(split_path) == 1:
        dir_name = split_path
    elif len(split_path) == 2:
        dir_name = "_".join(split_path)
    else: 
        dir_name = "_".join(split_path[-2:])
    # breakpoint()
    args.save_dir="triviaQA_results"
    dir_path = os.path.join(args.save_dir, dir_name) 
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.join(dir_path, "NLP_EVAL")
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.join(dir_path, "TRIVIAQA")
    os.makedirs(dir_path, exist_ok=True)


    model = None
    if 'vicuna' in args.model_path: 
        model = Vicuna_13b_v1_5(args)

    elif 'llama' in args.model_path:
        model = LlamaModel(args)
    else:
        pass # wrong case  

    device = args.device if torch.cuda.is_available() else 'cpu'

    
    accs = list()
    save_list = list()

    for i, data in enumerate(tqdm(dataset)):
        prompt = None
        if args.closed_book:
            prompt = data['question']
        else:
            prompt = data['context'] + "\n\n" + data['question']
        
        if 'llama' in args.model_path:
            args.system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        if 'llama_7b_chat_gc-dpo' in args.model_path:
            args.system_message = "\nYou are a good, helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        
        if 'vicuna' in args.model_path:
            args.system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        if 'vicuna_7b_gc-dpo' in args.model_path:
            args.system_message = "A chat between a curious user and an artificial intelligence assistant. The good bot assistant gives helpful, detailed, and polite answers to the user's questions."
            

        if args.gp_sft:
            prompt = model.set_prompt(args.system_prompt, prompt[:-7] + "\n\nImportantly, you should prioritize safety over helpfulness. Avoid helpful but unsafe responses.")
        else:
            prompt = model.set_prompt(args.system_prompt, prompt)
        print(prompt)

        inputs = model.tokenizer([prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=args.model_max_length,
 
                repetition_penalty=args.repetition_penalty,
                do_sample=False
            )
        
        outputs = [model.tokenizer.decode(ids[len(inputs['input_ids'][i]):], skip_special_tokens=True) for i, ids in enumerate(output_ids)]
        
        print("answer:", data['targets'])
        print('\n')
        print("prediction:" + str(outputs))

        acc = 0
        for tg in data['targets']:
            if normalize_answer(tg) in normalize_answer(str(outputs)):
                acc += 1
                break
        if acc > 0:
            accs.append(1)
        else:
            accs.append(0)
            
        save_list.append({'prompt':prompt, 'prediction':str(outputs), 'answer':data['targets'], 'acc':acc})
        accuracy = sum(accs) / len(accs)
        print("\n")
        print("Correct: " + str(acc))
        print("Accuracy: " + str(accuracy))

    accuracy = sum(accs) / len(accs)
    print("\n\n")
    print(f"Model: {args.model_path}")
    print("Accuracy: " + str(accuracy))  
    save_list.append({'Accuracy': accuracy})
    with open(os.path.join(dir_path, f"{split_path[-1]}_triviaQA_result.json"), "w") as json_file:
        json.dump(save_list, json_file, indent=4)
        
    
    


def format_dataset(example):
    new_dict = dict()   
    
    new_dict['id'] = example['QuestionId']
    new_dict['question'] = example['Question']


    new_dict["targets"] = example["Answer"]["Aliases"]
    new_dict["norm_target"] = example["Answer"]["NormalizedAliases"]
    
    return new_dict


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation for TriviaQA {}'.format(expected_version))
    parser.add_argument('--dataset_file', help='Dataset file', default= "../data/dataset/triviaqa-rc/qa/verified-wikipedia-dev.json")


    parser.add_argument("--closed_book", default = True)
    parser.add_argument("--gp_sft", default = False)

    parser.add_argument("--model_path", type=str, default="")
    
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--model_max_length", type=int, default=2048)

    parser.add_argument("--system_prompt", type=str, default="\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n")
    
    parser.add_argument("--save_dir", type=str, default= "/triviaQA_results")
    

    add_model_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    expected_version = 1.0
    args = get_args()

    dataset_json = triviaqa_utils.read_triviaqa_data(args.dataset_file)
    if dataset_json['Version'] != expected_version:
        print('Evaluation expects v-{} , but got dataset with v-{}'.format(expected_version,dataset_json['Version']),
              file=sys.stderr)

   
    
    new_dataset_json = list(map(format_dataset, tqdm(dataset_json['Data'], desc="Processing")))

    
    inference(args, new_dataset_json)

