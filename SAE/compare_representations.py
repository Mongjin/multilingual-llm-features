import torch
import argparse
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import pearsonr
import numpy as np
import json
import random
from transformer_lens import HookedTransformer, utils

# --- Functions for similarity and correlation modes ---

def load_dataset(dataset_path):
    """Loads a dataset from a JSONL file and returns a dictionary mapping type to a list of texts."""
    print(f"Loading dataset from: {dataset_path}")
    dataset = pd.read_json(dataset_path, lines=True)
    type_to_texts = defaultdict(list)
    for _, row in dataset.iterrows():
        type_to_texts[row['type']].append(row['text'])
    return type_to_texts

def get_hidden_states(text, model, tokenizer, device):
    """Encodes text and returns hidden states from all layers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    avg_hidden_states = [state.mean(dim=1).squeeze() for state in hidden_states]
    return avg_hidden_states

def get_logits(text, model, tokenizer, device):
    """Encodes text and returns the logits for the next token."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[:, -1, :].squeeze()

def analyze_representation_output_similarity(model, tokenizer, data1, data2, common_types, num_layers, device, output_dir, model_name, lang_comparison_str):
    """Analyzes the relationship between hidden representation similarity and output similarity for each type."""
    print("Running representation vs. output similarity analysis (per type)...")
    # ... (rest of the function is the same as the original)

def get_lang_from_path(path):
    """Extracts a language code from a dataset path."""
    try:
        filename = os.path.basename(path)
        return filename.split('.')[0].split('_')[-1]
    except:
        return "lang"

# --- Function for deactivate_neurons mode (using TransformerLens) ---

def analyze_neuron_deactivation(model, dataset_path, neuron_analysis_results, output_dir, model_name, lan, module):
    module = module.upper()
    print(f"--- Starting Neuron Deactivation Analysis for language: {lan}, module: {module} ---")

    model_name_formatted = model_name.replace("/", "_")
    neuron_file_path = os.path.join(neuron_analysis_results, 'top_1_percent_neurons', model_name_formatted, f'top_1_percent_neurons_{module}.pth')

    if not os.path.exists(neuron_file_path):
        raise FileNotFoundError(f"Neuron file not found at: {neuron_file_path}")

    dataset = pd.read_json(dataset_path, lines=True)
    top_neurons_per_layer = torch.load(neuron_file_path)
    num_layers = model.cfg.n_layers

    def deactivation_hook(activation, hook, neurons_to_deactivate, is_attn=False):
        if is_attn:
            original_shape = activation.shape
            # Reshape from [batch, seq, n_heads, d_head] to [batch, seq, n_heads * d_head]
            reshaped_activation = activation.view(original_shape[0], original_shape[1], -1)
            # Deactivate in the flattened space
            reshaped_activation[:, :, neurons_to_deactivate] = 0
            # Reshape back to original
            return reshaped_activation.view(original_shape)
        else:
            # For MLP, shape is [batch, seq, d_mlp]
            activation[:, :, neurons_to_deactivate] = 0
            return activation

    top_neurons_hooks = []
    random_neurons_hooks = []
    is_attn_module = module in ['Q', 'K', 'V']

    for layer_idx, layer_neurons in enumerate(top_neurons_per_layer):
        if module == 'MLP':
            hook_point = utils.get_act_name("post", layer_idx, "mlp")
        else:  # Q, K, V
            hook_point = utils.get_act_name(module.lower(), layer_idx)

        if lan in layer_neurons and len(layer_neurons[lan]) > 0:
            top_neurons_for_lan = layer_neurons[lan].tolist()
            
            def hook_factory(neurons):
                # Closure to capture the correct neuron list and is_attn flag
                return lambda act, hook: deactivation_hook(act, hook, neurons, is_attn=is_attn_module)

            top_neurons_hooks.append((hook_point, hook_factory(top_neurons_for_lan)))

            num_top_neurons = len(top_neurons_for_lan)
            if is_attn_module:
                if module == 'Q':
                    num_total_neurons = model.cfg.d_head * model.cfg.n_heads
                else:  # K or V
                    num_total_neurons = model.cfg.d_head * model.cfg.n_key_value_heads
            else:  # MLP
                num_total_neurons = model.cfg.d_mlp
            
            random_neurons_indices = random.sample(range(num_total_neurons), min(num_top_neurons, num_total_neurons))
            random_neurons_hooks.append((hook_point, hook_factory(random_neurons_indices)))

    results = {
        'losses': {'original': [], f'top_1_percent_deactivated_{lan}_{module}': [], 'random_deactivated': []},
        'similarities': {
            f'top_1_percent_vs_original_{lan}_{module}': [[] for _ in range(num_layers + 1)],
            f'random_vs_original_{lan}_{module}': [[] for _ in range(num_layers + 1)]
        }
    }
    
    for text in tqdm(dataset['text'], desc=f"Analyzing texts for {lan} ({module})"):
        tokens = model.to_tokens(text)
        
        # 1. Original model run
        original_loss, original_cache = model.run_with_cache(tokens, loss_per_token=True)
        results['losses']['original'].append(original_loss.mean().item())

        # 2. Top 1% deactivated run
        for hook_point, hook_fn in top_neurons_hooks:
            model.add_hook(hook_point, hook_fn)
        top_loss, top_cache = model.run_with_cache(tokens, loss_per_token=True)
        model.reset_hooks()
        results['losses'][f'top_1_percent_deactivated_{lan}_{module}'].append(top_loss.mean().item())

        # 3. Random deactivated run
        for hook_point, hook_fn in random_neurons_hooks:
            model.add_hook(hook_point, hook_fn)
        random_loss, random_cache = model.run_with_cache(tokens, loss_per_token=True)
        model.reset_hooks()
        results['losses']['random_deactivated'].append(random_loss.mean().item())

        # 4. Calculate similarities
        for i in range(num_layers + 1):
            act_name = utils.get_act_name("resid_post", i - 1) if i > 0 else utils.get_act_name("embed")
            original_act = original_cache[act_name].mean(dim=1).squeeze()
            top_act = top_cache[act_name].mean(dim=1).squeeze()
            random_act = random_cache[act_name].mean(dim=1).squeeze()

            sim_top = F.cosine_similarity(original_act, top_act, dim=-1).item()
            sim_random = F.cosine_similarity(original_act, random_act, dim=-1).item()
            results['similarities'][f'top_1_percent_vs_original_{lan}_{module}'][i].append(sim_top)
            results['similarities'][f'random_vs_original_{lan}_{module}'][i].append(sim_random)

    avg_losses = {k: np.mean(v) for k, v in results['losses'].items()}
    avg_similarities = {k: [np.mean(layer_sims) for layer_sims in v] for k, v in results['similarities'].items()}
    final_results = {'average_losses': avg_losses, 'average_similarities': avg_similarities}

    results_path = os.path.join(output_dir, f"neuron_deactivation_results_{lan}_{module}.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Neuron deactivation results for {lan} ({module}) saved to {results_path}")

    labels = list(avg_losses.keys())
    values = list(avg_losses.values())
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.ylabel('Average Loss')
    plt.title(f'Causal Effect of Deactivating {module} Neurons for {lan.upper()} ({model_name})')
    plt.xticks(rotation=15, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom')
    plt.tight_layout()
    plot_path_loss = os.path.join(output_dir, f"neuron_deactivation_loss_comparison_{lan}_{module}.png")
    plt.savefig(plot_path_loss)
    print(f"Loss comparison plot for {lan} ({module}) saved to {plot_path_loss}")
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers + 1), avg_similarities[f'top_1_percent_vs_original_{lan}_{module}'], marker='o', linestyle='-', label=f'Top 1% ({lan.upper()}, {module}) Deactivated vs. Original')
    plt.plot(range(num_layers + 1), avg_similarities[f'random_vs_original_{lan}_{module}'], marker='x', linestyle='--', label='Random Deactivated vs. Original')
    plt.title(f'Hidden Representation Similarity after Deactivating {module} Neurons for {lan.upper()} ({model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.xticks(range(num_layers + 1))
    plt.legend()
    plt.tight_layout()
    plot_path_sim = os.path.join(output_dir, f"neuron_deactivation_similarity_comparison_{lan}_{module}.png")
    plt.savefig(plot_path_sim)
    print(f"Similarity comparison plot for {lan} ({module}) saved to {plot_path_sim}")
    plt.close()

def analyze_intervention(model, dataset1_path, dataset2_path, neuron_analysis_results, output_dir, model_name, module, lan):
    module = module.upper()
    print(f"--- Starting Intervention Analysis for module: {module} ---")

    model_name_formatted = model_name.replace("/", "_")
    neuron_file_path = os.path.join(neuron_analysis_results, 'top_1_percent_neurons', model_name_formatted, f'top_1_percent_neurons_{module}.pth')

    if not os.path.exists(neuron_file_path):
        raise FileNotFoundError(f"Neuron file not found at: {neuron_file_path}")

    data1 = load_dataset(dataset1_path)
    data2 = load_dataset(dataset2_path)
    common_types = sorted(list(set(data1.keys()) & set(data2.keys())))
    print(f"Found {len(common_types)} common sentence types between the two datasets.")

    top_neurons_per_layer = torch.load(neuron_file_path)
    num_layers = model.cfg.n_layers

    def deactivation_hook(activation, hook, neurons_to_deactivate, is_attn=False):
        if is_attn:
            original_shape = activation.shape
            reshaped_activation = activation.view(original_shape[0], original_shape[1], -1)
            reshaped_activation[:, :, neurons_to_deactivate] = 0
            return reshaped_activation.view(original_shape)
        else:
            activation[:, :, neurons_to_deactivate] = 0
            return activation

    results = {
        'clean_vs_clean': {},
        'clean_vs_top_deactivated': {},
        'clean_vs_random_deactivated': {}
    }

    for sent_type in tqdm(common_types, desc="Comparing Sentences under Intervention"):
        top_hooks = []
        random_hooks = []
        is_attn_module = module in ['Q', 'K', 'V']

        for layer_idx, layer_neurons in enumerate(top_neurons_per_layer):
            if module == 'MLP':
                hook_point = utils.get_act_name("post", layer_idx, "mlp")
            else:
                hook_point = utils.get_act_name(module.lower(), layer_idx)
            
            neuron_type = sent_type if lan is None else lan
            if neuron_type in layer_neurons and len(layer_neurons[neuron_type]) > 0:
                top_neurons_for_lan = layer_neurons[neuron_type].tolist()
                
                def hook_factory(neurons):
                    return lambda act, hook: deactivation_hook(act, hook, neurons, is_attn=is_attn_module)

                top_hooks.append((hook_point, hook_factory(top_neurons_for_lan)))

                num_top_neurons = len(top_neurons_for_lan)
                if is_attn_module:
                    if module == 'Q':
                        num_total_neurons = model.cfg.d_head * model.cfg.n_heads
                    else:  # K or V
                        num_total_neurons = model.cfg.d_head * model.cfg.n_key_value_heads
                else:  # MLP
                    num_total_neurons = model.cfg.d_mlp
                
                random_neurons_indices = random.sample(range(num_total_neurons), min(num_top_neurons, num_total_neurons))
                random_hooks.append((hook_point, hook_factory(random_neurons_indices)))

        list1 = data1[sent_type]
        list2 = data2[sent_type]
        if not list1 or not list2: continue

        all_sims_clean = []
        all_sims_top = []
        all_sims_random = []

        for text1, text2 in zip(list1, list2):
            if not text1 or not text2: continue
            try:
                tokens1 = model.to_tokens(text1)
                tokens2 = model.to_tokens(text2)

                tokens1 = tokens1.to(model.cfg.device)
                tokens2 = tokens2.to(model.cfg.device)

                # 1. Clean runs
                _, cache1_clean = model.run_with_cache(tokens1)
                _, cache2_clean = model.run_with_cache(tokens2)

                # 2. Top neuron deactivation on text2
                for hook_point, hook_fn in top_hooks:
                    model.add_hook(hook_point, hook_fn)
                _, cache2_top = model.run_with_cache(tokens2)
                model.reset_hooks()

                # 3. Random neuron deactivation on text2
                for hook_point, hook_fn in random_hooks:
                    model.add_hook(hook_point, hook_fn)
                _, cache2_rand = model.run_with_cache(tokens2)
                model.reset_hooks()

                # 4. Calculate similarities
                sims_clean = []
                sims_top = []
                sims_random = []
                for i in range(num_layers + 1):
                    act_name = utils.get_act_name("resid_post", i - 1) if i > 0 else utils.get_act_name("embed")
                    act1_clean = cache1_clean[act_name].to('cpu').mean(dim=1).squeeze()
                    act2_clean = cache2_clean[act_name].to('cpu').mean(dim=1).squeeze()
                    act2_top = cache2_top[act_name].to('cpu').mean(dim=1).squeeze()
                    act2_rand = cache2_rand[act_name].to('cpu').mean(dim=1).squeeze()

                    sims_clean.append(F.cosine_similarity(act1_clean.to('cpu'), act2_clean.to('cpu'), dim=-1).item())
                    sims_top.append(F.cosine_similarity(act1_clean.to('cpu'), act2_top.to('cpu'), dim=-1).item())
                    sims_random.append(F.cosine_similarity(act1_clean.to('cpu'), act2_rand.to('cpu'), dim=-1).item())
                
                all_sims_clean.append(sims_clean)
                all_sims_top.append(sims_top)
                all_sims_random.append(sims_random)

            except Exception as e:
                print(f"Skipping a sentence pair due to error: {e}")
                model.reset_hooks()
                continue
        
        if all_sims_clean:
            results['clean_vs_clean'][sent_type] = torch.tensor(all_sims_clean).mean(dim=0).tolist()
        if all_sims_top:
            results['clean_vs_top_deactivated'][sent_type] = torch.tensor(all_sims_top).mean(dim=0).tolist()
        if all_sims_random:
            results['clean_vs_random_deactivated'][sent_type] = torch.tensor(all_sims_random).mean(dim=0).tolist()

    lang1 = get_lang_from_path(dataset1_path)
    lang2 = get_lang_from_path(dataset2_path)
    lang_comparison_str = f"{lang1} vs {lang2}"

    # Save results to JSON
    if lan is None:
        results_path = os.path.join(output_dir, f"intervention_similarity_results_{lang_comparison_str}_{module}.json")
    else:
        results_path = os.path.join(output_dir, f"intervention_similarity_results_{lang_comparison_str}_{lan}_{module}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Intervention similarity results saved to {results_path}")

    # Plotting
    for sent_type in common_types:
        if sent_type not in results['clean_vs_clean']: continue
        
        plt.figure(figsize=(15, 10))
        
        plt.plot(range(num_layers + 1), results['clean_vs_clean'][sent_type], marker='o', linestyle='-', label=f'Clean {lang1} vs Clean {lang2}')
        if sent_type in results['clean_vs_top_deactivated']:
            plt.plot(range(num_layers + 1), results['clean_vs_top_deactivated'][sent_type], marker='x', linestyle='--', label=f'Clean {lang1} vs Top Deactivated {lang2}')
        if sent_type in results['clean_vs_random_deactivated']:
            plt.plot(range(num_layers + 1), results['clean_vs_random_deactivated'][sent_type], marker='s', linestyle=':', label=f'Clean {lang1} vs Random Deactivated {lang2}')

        if lan is None:
            plt.title(f'Representation Similarity for Type \'{sent_type}\' under Intervention\n({lang_comparison_str}, {model_name}, {module})')
        else:
            plt.title(f'Representation Similarity for Type \'{sent_type}\' under Intervention\n({lang_comparison_str}, {model_name}, {lan},{module})')
        plt.xlabel('Layer')
        plt.ylabel('Cosine Similarity')
        plt.grid(True)
        plt.xticks(range(num_layers + 1))
        plt.legend(title="Comparison Type")
        plt.tight_layout()
        if lan is None:
            plot_path = os.path.join(output_dir, f"intervention_similarity_{sent_type}_{module}.png")
        else:
            plot_path = os.path.join(output_dir, f"intervention_{lan}_similarity_{sent_type}_{module}.png")
        plt.savefig(plot_path)
        print(f"Plot for sentence type \'{sent_type}, interventing {module}\' saved to {plot_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare hidden representations or analyze neuron deactivation of an LLM.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model to analyze.")
    parser.add_argument("--dataset1_path", type=str, default="./data/multitask_en.jsonl", help="Path to the first dataset.")
    parser.add_argument("--dataset2_path", type=str, default="./data/multitask_kor.jsonl", help="Path to the second dataset.")
    parser.add_argument("--output_dir", type=str, default="./representation_comparison", help="Directory to save the results.")
    parser.add_argument("--mode", type=str, choices=['similarity', 'correlation', 'deactivate_neurons', 'intervention'], default='similarity', help="Analysis mode to run.")
    parser.add_argument("--neuron_analysis_results", type=str, default="neuron_analysis_results", help="Directory containing the neuron analysis results for deactivation/intervention mode.")
    parser.add_argument("--lan", type=str, help="Language to target for neuron deactivation/intervention.")
    parser.add_argument("--module", type=str, choices=['mlp', 'q', 'k', 'v'], help="Module to target for neuron deactivation/intervention.")

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name_formatted = args.model_name.replace("/", "_")
    model_output_dir = os.path.join(args.output_dir, model_name_formatted)
    os.makedirs(model_output_dir, exist_ok=True)

    if args.mode in ['deactivate_neurons', 'intervention']:
        print(f"Loading model with TransformerLens: {args.model_name}")
        n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        model = HookedTransformer.from_pretrained(args.model_name, device=device, n_devices=n_devices)
        model.eval()

        if args.mode == 'deactivate_neurons':
            if not args.lan or not args.module:
                raise ValueError("--lan and --module are required for this mode.")
            analyze_neuron_deactivation(model, args.dataset1_path, args.neuron_analysis_results, model_output_dir, args.model_name, args.lan, args.module)
        elif args.mode == 'intervention':
            if not args.lan or not args.module:
                args.lan = None
            analyze_intervention(model, args.dataset1_path, args.dataset2_path, args.neuron_analysis_results, model_output_dir, args.model_name, args.module, args.lan)

    else: # similarity or correlation
        print(f"Loading model with AutoModelForCausalLM: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model.eval()

        lang1 = get_lang_from_path(args.dataset1_path)
        lang2 = get_lang_from_path(args.dataset2_path)
        lang_comparison_str = f"{lang1} vs {lang2}"
        
        data1 = load_dataset(args.dataset1_path)
        data2 = load_dataset(args.dataset2_path)
        common_types = sorted(list(set(data1.keys()) & set(data2.keys())))
        print(f"Found {len(common_types)} common sentence types between the two datasets.")
        
        num_layers = model.config.num_hidden_layers + 1

        if args.mode == 'correlation':
            analyze_representation_output_similarity(model, tokenizer, data1, data2, common_types, num_layers, device, model_output_dir, args.model_name, lang_comparison_str)
        
        elif args.mode == 'similarity':
            type_similarities = {}
            for sent_type in tqdm(common_types, desc="Comparing Sentences"):
                list1 = data1[sent_type]
                list2 = data2[sent_type]
                if not list1 or not list2: continue

                all_pair_similarities = []
                for text1, text2 in zip(list1, list2):
                    if not text1 or not text2: continue
                    try:
                        hidden_states1 = get_hidden_states(text1, model, tokenizer, device)
                        hidden_states2 = get_hidden_states(text2, model, tokenizer, device)
                        similarities = [F.cosine_similarity(h1, h2, dim=0).item() for h1, h2 in zip(hidden_states1, hidden_states2)]
                        all_pair_similarities.append(similarities)
                    except Exception as e:
                        print(f"Skipping a sentence pair due to error: {e}")
                        continue
                
                if all_pair_similarities:
                    type_similarities[sent_type] = torch.tensor(all_pair_similarities).mean(dim=0).tolist()

            results_path = os.path.join(model_output_dir, "similarity_results.jsonl")
            with open(results_path, 'w') as f:
                for sent_type, similarities in type_similarities.items():
                    f.write(json.dumps({
                        "type": sent_type,
                        "model_name": args.model_name,
                        "languages": lang_comparison_str,
                        "layer_similarities": similarities
                    }) + '\n')
            print(f"Similarity results saved to {results_path}")

            plt.figure(figsize=(15, 10))
            for sent_type, similarities in type_similarities.items():
                plt.plot(range(num_layers), similarities, marker='o', linestyle='-', label=sent_type)
            plt.title(f'Representation Similarity Across Layers ({lang_comparison_str}, {args.model_name})')
            plt.xlabel('Layer')
            plt.ylabel('Cosine Similarity')
            plt.grid(True)
            plt.xticks(range(num_layers))
            plt.legend(title="Sentence Type", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plot_path = os.path.join(model_output_dir, "cross_lingual_similarity_by_type.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()