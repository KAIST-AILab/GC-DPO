import argparse

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from peft import PeftModel, get_peft_model


class Vicuna_13b_v1_5():
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model, self.tokenizer = self._load_model(self.args)
    
    def _load_model(self, args):
        model, tokenizer = load_model(
            args.model_path,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory, # None
            dtype = torch.float16,
            load_8bit=args.load_8bit, # False
            cpu_offloading=args.cpu_offloading, # False
            revision=args.revision,  # main
            debug=args.debug # False
        )
        tokenizer.model_max_length = 2048
        
        #model = model.merge_and_unload()

        return model, tokenizer
    
    def _load_model_new(self, args):

        compute_dtype = (
        torch.float16
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        )

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
            if args.q_lora
            else None,
        )

        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        breakpoint()
        model = model.merge_andunload()
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
            #trust_remote_code=model_args.trust_remote_code,
        )

        return model, tokenizer

    


    def set_prompt(self, system_prompt, user_prompt):
        conv = get_conversation_template(self.args.model_path)
        # breakpoint()
        # Default system message is "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        conv.system_message = conv.system_message if system_prompt=="default" else system_prompt
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt
    

    def inference(self, system_prompt, user_prompt, temperature, repetition_penalty, max_new_tokens):
        prompt = self.set_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.args.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        return outputs

    def inference_batch(self, system_prompts, user_prompts, temperature, repetition_penalty, max_new_tokens):
        # Generating multiple prompts for the batch
        prompts = [self.set_prompt(system_prompt, user_prompt) for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
        
        # Tokenizing the batch of prompts
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.args.device)
        
        with torch.no_grad():  # Disable gradient computation for inference speedup
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 1e-5 else False
            )
        
        # Decoding the generated tokens to strings
        # outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        outputs = [self.tokenizer.decode(ids[len(inputs['input_ids'][i]):], skip_special_tokens=True) for i, ids in enumerate(output_ids)]
        return outputs
    
    def inference_batch_embedding(self, system_prompts, user_prompts, temperature, repetition_penalty, max_new_tokens):
        # Generating multiple prompts for the batch
        prompts = [self.set_prompt(system_prompt, user_prompt) for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
        
        # Tokenizing the batch of prompts
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.args.device)
        
        with torch.no_grad(): 
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=True if temperature > 1e-5 else False
            )
        
        
        hidden_states = output_ids.hidden_states[-1]
        
        outputs = [self.tokenizer.decode(ids[len(inputs['input_ids'][i]):], skip_special_tokens=True) for i, ids in enumerate(output_ids[0])]
        
        return outputs, hidden_states

class LlamaModel():
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model, self.tokenizer = self._load_model(self.args)
    
    def _load_model(self, args):
        
        tokenizer = AutoTokenizer.from_pretrained(
            # args.model_path
            "meta-llama/Llama-2-7b-chat-hf"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                load_in_8bit = args.load_8bit, # False
                torch_dtype = torch.bfloat16 #torch.bfloat16 
            ).to(self.args.device)


        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        if model_vocab_size != tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            base_model.resize_token_embeddings(tokenzier_vocab_size)

        return base_model, tokenizer

    
    def set_prompt(self, system_prompt, user_prompt):
        model_name = "llama-2" if "Llama-2" in self.args.model_path else "vicuna"
        conv = get_conversation_template(model_name)
        conv.system_message = system_prompt
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # breakpoint()
        return prompt
    
    def inference_batch(self, system_prompts, user_prompts, temperature, repetition_penalty, max_length):
        # Generating multiple prompts for the batch
        prompts = None
        if system_prompts == None:
            prompts = [user_prompt for user_prompt in user_prompts]
        else:
            prompts = [self.set_prompt(system_prompt, user_prompt) for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
      
        inputs = self.tokenizer(text=prompts, return_tensors="pt").to(self.args.device)

        
        with torch.no_grad():  
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 1e-5 else False
            )
        
        outputs = [self.tokenizer.decode(ids[len(inputs['input_ids'][i]):], skip_special_tokens=True) for i, ids in enumerate(output_ids)]


        return outputs