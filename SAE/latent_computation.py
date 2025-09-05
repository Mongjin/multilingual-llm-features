import torch
import numpy as np
from transformers import AutoModelForCausalLM,  AutoTokenizer
import os
import json
import pandas as pd
from tqdm import tqdm
from utils import load_args,load_sae

torch.set_grad_enabled(False)  # avoid blowing up mem

def gather_residual_activations(model, target_layer, inputs):
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs
    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act


# Calculate the per-token, per-layer SAE activation values across the given dataset.
def hf_model_gen(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto',torch_dtype="auto",)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    multilingual_data = pd.read_json(args.feature_data_path, lines=True)
    # multilingual_data = pd.read_json('./data/metamath_thinking.jsonl', lines=True)

    for layer in tqdm(range(model.config.num_hidden_layers)):
        sae = load_sae(layer,args)
        all_sae_acts = []
        for prompt in tqdm(multilingual_data['text'].tolist()):
            # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
            inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
            target_act = gather_residual_activations(model, layer, inputs)
            if 'Llama' in args.model:
                sae_acts = sae.encode(target_act.to(torch.bfloat16)).cpu().to(torch.float32)
            else:
                sae_acts = sae.encode(target_act.to(torch.float32)).cpu()
            all_sae_acts.append(sae_acts)
        
        save_dir = f'{args.sae_output_dir}/{args.model}/layer_{layer}/'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_sae_acts, os.path.join(save_dir, 'sae_acts.pth'))


if __name__ == "__main__":
    args=load_args()
    hf_model_gen(args)