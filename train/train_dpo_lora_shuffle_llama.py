
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import torch
import hashlib

from fastchat.model import get_conversation_template

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

@dataclass
class DPOTrainingArguments(DPOConfig): 
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    beta: float = 0.1
    flash_attn: bool = False
    bf16: bool = True 

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def system_prompt_perturbation(system_prompt, perturbation_method, model_path="Llama-2-7b-chat-hf"):
    if "Llama-2-7b-chat-hf" in model_path:
        if perturbation_method == "none":
            # return system_prompt
            return "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        elif perturbation_method == "classic":
            return "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        elif perturbation_method == "minimal_perturbation_llama": 
            mapper = {
                "You are a good bot assistant.": "\nYou are a good, helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n",
                "You are a bad bot assistant.": "\nYou are a bad, helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
            }
            return mapper[system_prompt]
        else:
            raise ValueError(f"Invalid perturbation method: {perturbation_method}")
    elif "Llama-2-7b-hf" in model_path:
        if perturbation_method == "none":
            # return system_prompt
            return ""
        elif perturbation_method == "classic":
            return ""
        elif perturbation_method == "minimal_perturbation_llama": 
            mapper = {
                "You are a good bot assistant.": "You are a good, helpful, respectful and honest assistant.",
                "You are a bad bot assistant.": "You are a bad, helpful, respectful and honest assistant."
            }
            return mapper[system_prompt]
        else:
            raise ValueError(f"Invalid perturbation method: {perturbation_method}")
    else:
        raise ValueError(f"Invalid model path for system prompt perturbation: {model_path}")

############################################################################################################
def goal_dpo_data_format(example):
    conv = get_conversation_template("llama-2")
    # breakpoint()
    if "sha256" in dpo_training_args.output_dir:
        conv.system_message = system_prompt_perturbation(example['goal'], perturbation_method="sha256")
    elif "mapping" in dpo_training_args.output_dir:
        conv.system_message = system_prompt_perturbation(example['goal'], perturbation_method="mapping")
    elif "none" in dpo_training_args.output_dir:
        conv.system_message = system_prompt_perturbation(example['goal'], perturbation_method="none")
    elif "classic" in dpo_training_args.output_dir:
        conv.system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    elif "minimal_perturbation" in dpo_training_args.output_dir:
        conv.system_message = system_prompt_perturbation(example['goal'], perturbation_method="minimal_perturbation_llama")
    # breakpoint()
    # conv.system_message = system_prompt_perturbation(example['goal'], perturbation_method=perturbation_method_wrapper)
    conv.append_message(conv.roles[0], example['prompt']) # user
    conv.append_message(conv.roles[1], None) # assistant
    prompt = conv.get_prompt()
    # breakpoint()
    chosen = example['chosen']
    rejected = example['rejected']
    # breakpoint()
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
############################################################################################################

def train():
    global dpo_training_args
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DPOTrainingArguments, LoraArguments)
    )
    (model_args, data_args, dpo_training_args, lora_args) = parser.parse_args_into_dataclasses() 

    
    if dpo_training_args.flash_attn:
        replace_llama_attn_with_flash_attn()


    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(dpo_training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.bfloat16
        # if dpo_training_args.fp16
        # else (torch.bfloat16 if dpo_training_args.bf16 else torch.float32)
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    # breakpoint()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=dpo_training_args.cache_dir,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
    )


    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=dpo_training_args.cache_dir,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=dpo_training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)
    ref_model = get_peft_model(ref_model, lora_config)

    
    if dpo_training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)

    if dpo_training_args.deepspeed is not None and dpo_training_args.local_rank == 0:
        model.print_trainable_parameters()

    if dpo_training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=dpo_training_args.cache_dir,
        model_max_length=dpo_training_args.model_max_length,
        padding_side=model_args.padding_side, # "right"
        use_fast=False,
        #trust_remote_code=model_args.trust_remote_code,
    )


    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    
    dpo_dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    dpo_dataset = dpo_dataset.shuffle(seed=1)
    original_columns = dpo_dataset.column_names
    

    train_dataset = dpo_dataset.map(
        function =goal_dpo_data_format,
        remove_columns = original_columns
    )

    #breakpoint()

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=dpo_training_args,
        train_dataset= train_dataset, 
        max_length=dpo_training_args.model_max_length,
        tokenizer= tokenizer,
    )

    model.config.use_cache = False
    
    if list(pathlib.Path(dpo_training_args.output_dir).glob("checkpoint-*")):
        dpo_trainer.train(resume_from_checkpoint=True)
    else:
        dpo_trainer.train()
        

    dpo_trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = dpo_trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if dpo_training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if dpo_training_args.local_rank == 0:
        model.save_pretrained(dpo_training_args.output_dir, state_dict=state_dict)

if __name__ == "__main__":
    train()