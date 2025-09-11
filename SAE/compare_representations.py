
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
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    # Average hidden states across the sequence length for each layer
    avg_hidden_states = [state.mean(dim=1).squeeze() for state in hidden_states]
    return avg_hidden_states

def get_logits(text, model, tokenizer, device):
    """Encodes text and returns the logits for the next token."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    # Return the logits for the last token in the sequence
    return outputs.logits[:, -1, :].squeeze()

def analyze_representation_output_similarity(model, tokenizer, data1, data2, common_types, num_layers, device, output_dir, model_name, lang_comparison_str):
    """Analyzes the relationship between hidden representation similarity and output similarity for each type."""
    print("Running representation vs. output similarity analysis (per type)...")

    type_layer_hidden_sims = {sent_type: [[] for _ in range(num_layers)] for sent_type in common_types}
    type_output_sims = {sent_type: [] for sent_type in common_types}

    for sent_type in tqdm(common_types, desc="Analyzing Similarity Correlation"):
        list1 = data1[sent_type]
        list2 = data2[sent_type]

        for text1, text2 in zip(list1, list2):
            if not text1 or not text2:
                continue
            try:
                hidden_states1 = get_hidden_states(text1, model, tokenizer, device)
                hidden_states2 = get_hidden_states(text2, model, tokenizer, device)
                logits1 = get_logits(text1, model, tokenizer, device)
                logits2 = get_logits(text2, model, tokenizer, device)

                output_sim = F.cosine_similarity(logits1, logits2, dim=0).item()
                type_output_sims[sent_type].append(output_sim)

                for i in range(num_layers):
                    hidden_sim = F.cosine_similarity(hidden_states1[i], hidden_states2[i], dim=0).item()
                    type_layer_hidden_sims[sent_type][i].append(hidden_sim)

            except Exception as e:
                print(f"Skipping a sentence pair due to error: {e}")
                continue

    # Calculate correlations and save results
    results_to_save = []
    type_correlations = {}
    for sent_type in common_types:
        correlations = []
        for i in range(num_layers):
            if len(type_layer_hidden_sims[sent_type][i]) > 1 and len(type_output_sims[sent_type]) > 1:
                corr, _ = pearsonr(type_layer_hidden_sims[sent_type][i], type_output_sims[sent_type])
                correlations.append(corr)
            else:
                correlations.append(float('nan'))
        type_correlations[sent_type] = correlations

        results_to_save.append({
            "type": sent_type,
            "model_name": model_name,
            "languages": lang_comparison_str,
            "layer_correlations": correlations,
            "layer_hidden_similarities": type_layer_hidden_sims[sent_type],
            "output_similarities": type_output_sims[sent_type]
        })

    results_path = os.path.join(output_dir, "correlation_results.jsonl")
    with open(results_path, 'w') as f:
        for item in results_to_save:
            f.write(json.dumps(item) + '\n')
    print(f"Correlation results saved to {results_path}")

    # Plotting the correlation per layer for each type
    plt.figure(figsize=(15, 10))
    for sent_type, correlations in type_correlations.items():
        plt.plot(range(num_layers), correlations, marker='o', linestyle='-', label=sent_type)
    
    plt.title(f'Correlation between Hidden Rep. and Output Similarity ({lang_comparison_str}, {model_name})')
    plt.xlabel('Layer')
    plt.ylabel('Pearson Correlation')
    plt.grid(True)
    plt.xticks(range(num_layers))
    plt.legend(title="Sentence Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    corr_plot_path = os.path.join(output_dir, "correlation_by_layer_per_type.png")
    plt.savefig(corr_plot_path)
    print(f"Correlation plot saved to {corr_plot_path}")
    plt.close()

    # Scatter plot for the last layer, colored by type
    plt.figure(figsize=(12, 8))
    for sent_type in common_types:
        plt.scatter(type_layer_hidden_sims[sent_type][-1], type_output_sims[sent_type], alpha=0.5, label=sent_type)
    
    plt.title(f'Last Layer Hidden Similarity vs. Output Similarity ({lang_comparison_str}, {model_name})')
    plt.xlabel('Hidden Representation Similarity (Cosine)')
    plt.ylabel('Output Logit Similarity (Cosine)')
    plt.grid(True)
    plt.legend(title="Sentence Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    scatter_plot_path = os.path.join(output_dir, "last_layer_scatter_per_type.png")
    plt.savefig(scatter_plot_path)
    print(f"Scatter plot saved to {scatter_plot_path}")
    plt.close()

def get_lang_from_path(path):
    """Extracts a language code from a dataset path (e.g., 'multitask_en.jsonl' -> 'en')."""
    try:
        filename = os.path.basename(path)
        # Assumes format like '..._en.jsonl'
        return filename.split('.')[0].split('_')[-1]
    except:
        return "lang" # Fallback

def main():
    parser = argparse.ArgumentParser(description="Compare hidden representations of an LLM for semantically equivalent sentences in different languages.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model to analyze.")
    parser.add_argument("--dataset1_path", type=str, default="./data/multitask_en.jsonl", help="Path to the first dataset.")
    parser.add_argument("--dataset2_path", type=str, default="./data/multitask_kor.jsonl", help="Path to the second dataset.")
    parser.add_argument("--output_dir", type=str, default="./representation_comparison", help="Directory to save the plot.")
    parser.add_argument("--run_correlation_analysis", action="store_true", help="Run the advanced correlation analysis.")
    args = parser.parse_args()

    if not args.run_correlation_analysis:
        torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a model-specific output directory
    model_name_formatted = args.model_name.replace("/", "_")
    model_output_dir = os.path.join(args.output_dir, model_name_formatted)
    os.makedirs(model_output_dir, exist_ok=True)

    # Get language comparison string for plot titles
    lang1 = get_lang_from_path(args.dataset1_path)
    lang2 = get_lang_from_path(args.dataset2_path)
    lang_comparison_str = f"{lang1} vs {lang2}"
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()

    # Load datasets
    data1 = load_dataset(args.dataset1_path)
    data2 = load_dataset(args.dataset2_path)

    # Find common types
    common_types = sorted(list(set(data1.keys()) & set(data2.keys())))
    print(f"Found {len(common_types)} common sentence types between the two datasets.")

    num_layers = model.config.num_hidden_layers + 1  # Including embedding layer

    if args.run_correlation_analysis:
        analyze_representation_output_similarity(model, tokenizer, data1, data2, common_types, num_layers, device, model_output_dir, args.model_name, lang_comparison_str)
    else:
        type_similarities = {}

        for sent_type in tqdm(common_types, desc="Comparing Sentences"):
            list1 = data1[sent_type]
            list2 = data2[sent_type]

            if not list1 or not list2:
                continue

            all_pair_similarities = []

            for text1, text2 in zip(list1, list2):
                if not text1 or not text2:
                    continue
                try:
                    hidden_states1 = get_hidden_states(text1, model, tokenizer, device)
                    hidden_states2 = get_hidden_states(text2, model, tokenizer, device)

                    similarities = []
                    for i in range(num_layers):
                        similarity = F.cosine_similarity(hidden_states1[i], hidden_states2[i], dim=0)
                        similarities.append(similarity.item())
                    all_pair_similarities.append(similarities)
                except Exception as e:
                    print(f"Skipping a sentence pair due to error: {e}")
                    continue
            
            if all_pair_similarities:
                avg_similarities = torch.tensor(all_pair_similarities).mean(dim=0).tolist()
                type_similarities[sent_type] = avg_similarities

        # Save results to JSONL
        results_to_save = []
        for sent_type, similarities in type_similarities.items():
            results_to_save.append({
                "type": sent_type,
                "model_name": args.model_name,
                "languages": lang_comparison_str,
                "layer_similarities": similarities
            })
        
        results_path = os.path.join(model_output_dir, "similarity_results.jsonl")
        with open(results_path, 'w') as f:
            for item in results_to_save:
                f.write(json.dumps(item) + '\n')
        print(f"Similarity results saved to {results_path}")

        # Plotting
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
