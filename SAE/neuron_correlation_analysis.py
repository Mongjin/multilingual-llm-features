import torch
import argparse
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def get_top_1_percent_neurons(analysis_dir, module):
    """
    Performs a two-pass analysis to identify the top 1% of language-specific neurons for a given dataset.
    It self-determines the languages present in the dataset.
    """
    print(f"\nStarting Top 1% analysis for {analysis_dir}...")
    file_pattern = f"layer_*_top_neurons_{module}.pth"
    analysis_files = sorted(glob.glob(os.path.join(analysis_dir, file_pattern)))

    if not analysis_files:
        raise FileNotFoundError(f"No analysis files found in {analysis_dir} with pattern {file_pattern}")

    languages = torch.load(analysis_files[0])['languages']
    print(f"Found languages in {analysis_dir}: {languages}")

    num_layers = len(analysis_files)

    if module == 'mlp':
        all_scores_per_lang = {lang: [] for lang in languages}
    else: # attn
        all_scores_per_lang = {c: {lang: [] for lang in languages} for c in ['q', 'k', 'v']}

    print("Pass 1: Gathering scores to determine global thresholds...")
    for file_path in tqdm(analysis_files, desc=f"Gathering Scores from {os.path.basename(analysis_dir)}"):
        data = torch.load(file_path)
        if module == 'mlp':
            scores = data['all_scores']
            for i, lang in enumerate(languages):
                all_scores_per_lang[lang].append(scores[i])
        else: # attn
            for comp in ['q', 'k', 'v']:
                data_key = f'avg_activations_{comp}'
                if data_key in data:
                    scores = data[data_key]['all_scores']
                    for i, lang in enumerate(languages):
                        all_scores_per_lang[comp][lang].append(scores[i])

    thresholds = {lang: 0 for lang in languages} if module == 'mlp' else {c: {lang: 0 for lang in languages} for c in ['q', 'k', 'v']}
    if module == 'mlp':
        for lang in languages:
            if all_scores_per_lang[lang]:
                lang_scores_tensor = torch.cat(all_scores_per_lang[lang])
                if lang_scores_tensor.numel() > 0:
                    thresholds[lang] = torch.quantile(lang_scores_tensor, 0.99)
    else: # attn
        for comp in ['q', 'k', 'v']:
            for lang in languages:
                if all_scores_per_lang[comp][lang]:
                    lang_scores_tensor = torch.cat(all_scores_per_lang[comp][lang])
                    if lang_scores_tensor.numel() > 0:
                        thresholds[comp][lang] = torch.quantile(lang_scores_tensor, 0.99)

    if module == 'mlp':
        top_neurons_by_layer = {layer: {lang: set() for lang in languages} for layer in range(num_layers)}
    else: # attn
        top_neurons_by_layer = {layer: {lang: {c: set() for c in ['q', 'k', 'v']} for lang in languages} for layer in range(num_layers)}

    print("Pass 2: Identifying top 1% neurons per layer...")
    for layer_idx, file_path in enumerate(tqdm(analysis_files, desc=f"Identifying Neurons in {os.path.basename(analysis_dir)}")):
        data = torch.load(file_path)
        if module == 'mlp':
            scores = data['all_scores']
            low_entropy_set = set(data['low_entropy_indices'].tolist())
            for i, lang in enumerate(languages):
                if lang in thresholds:
                    above_threshold_indices = torch.where(scores[i] > thresholds[lang])[0].tolist()
                    final_indices = {idx for idx in above_threshold_indices if idx in low_entropy_set}
                    top_neurons_by_layer[layer_idx][lang] = final_indices
        else: # attn
            for lang in languages:
                lang_idx = languages.index(lang)
                for comp in ['q', 'k', 'v']:
                    if lang in thresholds[comp]:
                        data_key = f'avg_activations_{comp}'
                        if data_key in data:
                            scores = data[data_key]['all_scores']
                            low_entropy_set = set(data[data_key]['low_entropy_indices'].tolist())
                            above_threshold_indices = torch.where(scores[lang_idx] > thresholds[comp][lang])[0].tolist()
                            final_indices = {idx for idx in above_threshold_indices if idx in low_entropy_set}
                            top_neurons_by_layer[layer_idx][lang][comp] = final_indices
    return top_neurons_by_layer, languages

def calculate_jaccard_similarity(set1, set2):
    if not isinstance(set1, set) or not isinstance(set2, set):
        return 0.0  # Return 0 similarity if inputs are not sets
    if not set1 and not set2:
        return 0.0 # Define Jaccard similarity for two empty sets as 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def analyze_correlation(args):
    model_name_safe = args.model_name.replace("/", "_")
    analysis_dir1 = os.path.join(args.analysis_results_dir1, model_name_safe)
    analysis_dir2 = os.path.join(args.analysis_results_dir2, model_name_safe)
    
    mode_str = f"cross_lingual_{args.lang1}_vs_{args.lang2}" if args.mode == 'cross' else "intra_lingual"
    output_subdir = os.path.join(args.plot_output_dir, f"{model_name_safe}_{args.module}_top1_percent_{mode_str}_with_counts")
    os.makedirs(output_subdir, exist_ok=True)

    print(f"Analyzing correlation for model: {args.model_name}")
    print(f"Plotting results to: {output_subdir}")

    top_neurons1, languages1 = get_top_1_percent_neurons(analysis_dir1, args.module)
    top_neurons2, languages2 = get_top_1_percent_neurons(analysis_dir2, args.module)

    common_layers = sorted(list(set(top_neurons1.keys()).intersection(set(top_neurons2.keys()))))

    if args.mode == 'cross':
        if args.lang1 not in languages1:
            raise ValueError(f"--lang1 '{args.lang1}' not found in dataset 1 languages: {languages1}")
        if args.lang2 not in languages2:
            raise ValueError(f"--lang2 '{args.lang2}' not found in dataset 2 languages: {languages2}")
        language_pairs = [(args.lang1, args.lang2)]
        print(f"\nRunning CROSS-LINGUAL analysis for {args.lang1} vs {args.lang2}...")
    else:
        common_languages = sorted(list(set(languages1).intersection(set(languages2))))
        if not common_languages:
            raise ValueError("No common languages found between the two datasets for intra-lingual analysis.")
        language_pairs = [(lang, lang) for lang in common_languages]
        print(f"\nRunning INTRA-LINGUAL analysis for common languages: {common_languages}...")

    plt.style.use('seaborn-v0_8-whitegrid')
    x_indices = np.arange(len(common_layers))

    for lang1, lang2 in tqdm(language_pairs, desc="Processing Language Pairs"):
        fig, ax1 = plt.subplots(figsize=(18, 9))
        ax2 = ax1.twinx()

        if args.module == 'mlp':
            plot_data = {'similarity': [], 'count1': [], 'count2': []}
            for layer in common_layers:
                set1 = top_neurons1.get(layer, {}).get(lang1, set())
                set2 = top_neurons2.get(layer, {}).get(lang2, set())
                plot_data['similarity'].append(calculate_jaccard_similarity(set1, set2))
                plot_data['count1'].append(len(set1))
                plot_data['count2'].append(len(set2))
            
            l1 = ax1.plot(x_indices, plot_data['similarity'], marker='o', linestyle='-', color='crimson', label='Jaccard Similarity')
            b1 = ax2.bar(x_indices - 0.2, plot_data['count1'], 0.4, alpha=0.6, color='steelblue', label=f'Count ({lang1.upper()})')
            b2 = ax2.bar(x_indices + 0.2, plot_data['count2'], 0.4, alpha=0.6, color='darkorange', label=f'Count ({lang2.upper()})')
            
            title_str = f'Top 1% MLP Neuron Overlap for {lang1.upper()} (General) vs. {lang2.upper()} (Math)'
            lns = l1
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper left')
            ax2.legend((b1, b2), (b1.get_label(), b2.get_label()), loc='upper right')

        else: # attn
            plot_data = {c: {'similarity': [], 'count1': [], 'count2': []} for c in ['q', 'k', 'v']}
            total_counts1, total_counts2 = [], []
            for layer in common_layers:
                total_set1, total_set2 = set(), set()
                for comp in ['q', 'k', 'v']:
                    set1 = top_neurons1.get(layer, {}).get(lang1, {}).get(comp, set())
                    set2 = top_neurons2.get(layer, {}).get(lang2, {}).get(comp, set())
                    plot_data[comp]['similarity'].append(calculate_jaccard_similarity(set1, set2))
                    total_set1.update(set1)
                    total_set2.update(set2)
                total_counts1.append(len(total_set1))
                total_counts2.append(len(total_set2))

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            lines = []
            for i, comp in enumerate(['q', 'k', 'v']):
                l = ax1.plot(x_indices, plot_data[comp]['similarity'], marker='o', linestyle='-', color=colors[i], label=f'Jaccard ({comp.upper()})')
                lines.extend(l)
            
            b1 = ax2.bar(x_indices - 0.2, total_counts1, 0.4, alpha=0.5, color='gray', label=f'Total Count ({lang1.upper()})')
            b2 = ax2.bar(x_indices + 0.2, total_counts2, 0.4, alpha=0.5, color='black', label=f'Total Count ({lang2.upper()})')
            
            title_str = f'Top 1% Attention Neuron Overlap for {lang1.upper()} (General) vs. {lang2.upper()} (Math)'
            labs = [l.get_label() for l in lines]
            ax1.legend(lines, labs, loc='upper left')
            ax2.legend((b1, b2), (b1.get_label(), b2.get_label()), loc='upper right')

        ax1.set_title(f'{title_str}\nModel: {args.model_name}', fontsize=16, weight='bold')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Jaccard Similarity', fontsize=12, color='crimson')
        ax1.tick_params(axis='y', labelcolor='crimson')
        ax1.set_ylim(0, 1)
        ax1.grid(True, which='major', axis='x')

        ax2.set_ylabel('Neuron Count', fontsize=12)
        ax2.set_ylim(0, max(ax2.get_ylim()[1], 1) * 1.1)

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(common_layers, rotation=45)
        fig.tight_layout()

        plot_filename = f'top1_neuron_correlation_{lang1}_vs_{lang2}_with_counts.png'
        save_path = os.path.join(output_subdir, plot_filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    print(f"\nCorrelation analysis complete. Plots saved in: {output_subdir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation of TOP 1% language-specific neurons.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model name.")
    parser.add_argument("--analysis_results_dir1", type=str, default="./neuron_analysis_results", help="Results dir for dataset 1 (general).")
    parser.add_argument("--analysis_results_dir2", type=str, default="./neuron_analysis_results_math", help="Results dir for dataset 2 (math).")
    parser.add_argument("--plot_output_dir", type=str, default="./plot/neuron_correlation", help="Base directory for plots.")
    parser.add_argument("--module", type=str, default="mlp", choices=["mlp", "attn"], help="Model module to analyze.")
    parser.add_argument("--mode", type=str, default="intra", choices=["intra", "cross"], help="Analysis mode: 'intra' for same-language, 'cross' for cross-language.")
    parser.add_argument("--lang1", type=str, default=None, help="Language from dataset 1 (e.g., 'en') for cross-lingual analysis.")
    parser.add_argument("--lang2", type=str, default=None, help="Language from dataset 2 (e.g., 'ko') for cross-lingual analysis.")
    
    args = parser.parse_args()

    if args.mode == 'cross' and (not args.lang1 or not args.lang2):
        parser.error("--lang1 and --lang2 are required for --mode='cross'")

    analyze_correlation(args)

if __name__ == "__main__":
    main()
