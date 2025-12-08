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


class VicunaModel():
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
            debug=False # False
        )
        tokenizer.model_max_length = 2048
        
        #model = model.merge_and_unload()

        return model, tokenizer

    def set_prompt(self, system_prompt, user_prompt):
        conv = get_conversation_template(self.args.model_path)
        conv.system_message = system_prompt
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

    def inference_batch(self, system_prompts, user_prompts, temperature, repetition_penalty, max_length):
        #Generating multiple prompts for the batch
        
        prompts = [self.set_prompt(system_prompt, user_prompt) for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
        
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.args.device)

        
        with torch.no_grad():  # Disable gradient computation for inference speedup
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 1e-5 else False
            )
        

        outputs = [self.tokenizer.decode(ids[len(inputs['input_ids'][i]):], skip_special_tokens=True) for i, ids in enumerate(output_ids)]
        
        return outputs
