import torch
import argparse
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from transformer_lens import HookedTransformer, utils
import heapq

# Suppress gradient calculations for inference
torch.set_grad_enabled(False)

def load_dataset(dataset_path, task_type):
    """Loads and filters a dataset from a JSONL file."""
    print(f"Loading and filtering dataset for type: {task_type}")
    dataset = pd.read_json(dataset_path, lines=True)
    filtered_dataset = dataset[dataset['type'] == task_type]
    if filtered_dataset.empty:
        raise ValueError(f"No data found for type '{task_type}' in {dataset_path}")
    print(f"Found {len(filtered_dataset)} texts for type '{task_type}'.")
    return filtered_dataset

def get_hook_point(module, layer_idx):
    """Get the appropriate hook point for the given module and layer."""
    if module.lower() == 'mlp':
        return utils.get_act_name("post", layer_idx)
    elif module.lower() in ['q', 'k', 'v']:
        return utils.get_act_name(module.lower(), layer_idx)
    else:
        raise ValueError(f"Unsupported module: {module}")

def analyze_activations_single_pass(model, dataset, modules_to_run, top_neurons_base_path, model_name_formatted, top_n_texts, neuron_type):
    """
    Analyzes neuron activations for multiple modules in a single pass over the dataset.
    """
    print("--- Starting Single-Pass Neuron Activation Analysis ---")

    # 1. Load all neuron data and prepare hooks
    top_neurons_all_modules = {}
    source_paths = {}
    all_hook_points = set()
    top_texts_per_module = {}

    for module in modules_to_run:
        neuron_file_name = f'top_1_percent_neurons_{module.upper()}.pth'
        top_neurons_path = os.path.join(top_neurons_base_path, neuron_file_name)

        if os.path.exists(top_neurons_path):
            print(f"Loading {module.upper()} top neurons from: {top_neurons_path}")
            top_neurons_data = torch.load(top_neurons_path)
            top_neurons_all_modules[module] = top_neurons_data
            source_paths[module] = top_neurons_path
            top_texts_per_module[module] = {}

            for layer_idx, _ in enumerate(top_neurons_data):
                all_hook_points.add(get_hook_point(module, layer_idx))
        else:
            print(f"\nWarning: {module.upper()} top neuron file not found at {top_neurons_path}. Skipping.")

    if not top_neurons_all_modules:
        print("No neuron data found for any specified modules. Exiting.")
        return {}, {}

    # 2. Initialize result structures (heaps)
    for module, top_neurons_data in top_neurons_all_modules.items():
        for layer_idx, layer_neurons_by_lang in enumerate(top_neurons_data):
            if neuron_type in layer_neurons_by_lang:
                neurons_for_lang = layer_neurons_by_lang[neuron_type].tolist()
                if neurons_for_lang:
                    top_texts_per_module[module][layer_idx] = {neuron_idx: [] for neuron_idx in neurons_for_lang}

    # 3. Single loop over the dataset
    print(f"\nProcessing {len(dataset)} texts with {len(all_hook_points)} hooks active...")
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Analyzing Texts"):
        text = row['text']
        if not text: continue

        tokens = model.to_tokens(text, truncate=True)
        
        try:
            _, cache = model.run_with_cache(tokens, names_filter=list(all_hook_points))

            # 4. Process the cache for each module
            for module, results_dict in top_texts_per_module.items():
                is_attn = module in ['q', 'k', 'v']
                for layer_idx, neurons_dict in results_dict.items():
                    hook_point = get_hook_point(module, layer_idx)
                    if hook_point not in cache: continue

                    activations = cache[hook_point][0]
                    if is_attn:
                        activations = activations.view(activations.shape[0], -1)
                    
                    max_activations, max_indices = torch.max(activations, dim=0)

                    for neuron_idx in neurons_dict.keys():
                        if neuron_idx < len(max_activations):
                            activation_value = max_activations[neuron_idx].item()
                            token_position = max_indices[neuron_idx].item()
                            token_id = tokens[0, token_position]
                            activating_token = model.to_string([token_id])

                            heap = neurons_dict[neuron_idx]
                            # Store (activation, token, full_text) for context if needed later
                            item_to_store = (activation_value, activating_token)

                            if len(heap) < top_n_texts:
                                heapq.heappush(heap, item_to_store)
                            else:
                                heapq.heappushpop(heap, item_to_store)
        except Exception as e:
            print(f"Skipping text due to error: {e}")
            continue

    return top_texts_per_module, source_paths

def plot_module_results(results, module, top_n_texts, output_dir, source_path, task_type, neuron_type):
    """Plots the results for a single module (MLP, Q, K, or V)."""
    print(f"\n--- Generating {module.upper()} Plots ---")
    plot_dir = os.path.join(output_dir, f"top_{top_n_texts}_tokens_{module.lower()}")
    os.makedirs(plot_dir, exist_ok=True)
    
    for layer_idx, neurons in tqdm(results.items(), desc=f"Generating {module.upper()} Plots"):
        layer_plot_dir = os.path.join(plot_dir, f"layer_{layer_idx}")
        os.makedirs(layer_plot_dir, exist_ok=True)

        for neuron_idx, heap in neurons.items():
            if not heap: continue
            
            sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
            activations = [item[0] for item in sorted_items]
            tokens = [item[1] for item in sorted_items]
            # print(f"Neuron {neuron_idx} in Layer {layer_idx} top tokens: {tokens}")

            # Escape dollar signs and remove replacement characters for clean plotting
            cleaned_tokens = [token.replace('$', '\\$').replace("\ufffd", "") for token in tokens]

            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = range(len(cleaned_tokens))
            bars = ax.barh(y_pos, activations, align='center', color='lightgreen')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cleaned_tokens, fontsize=12)
            ax.invert_yaxis()
            ax.set_xlabel('Max Activation Value')
            title = f'Top {top_n_texts} Activating Tokens ({task_type}) for {neuron_type.upper()} {module.upper()} Neuron {neuron_idx} in Layer {layer_idx}'
            ax.set_title(title)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2., f'{width:.4f}', va='center', ha='left')

            plt.suptitle(f"Source: {os.path.basename(source_path)}", y=0.02, fontsize=8, color='gray')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            plot_path = os.path.join(layer_plot_dir, f'neuron_{neuron_idx}_top_tokens.png')
            plt.savefig(plot_path)
            plt.close(fig)
    print(f"{module.upper()} plots saved in {plot_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze texts that maximally activate top neurons.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model to analyze.")
    parser.add_argument("--analysis_results_dir", required=True, type=str, help="Directory containing the neuron analysis results (e.g., 'neuron_analysis_results_en_kor').")
    parser.add_argument("--dataset_path", type=str, default="./data/multilingual_data.jsonl", help="Path to the JSONL dataset.")
    parser.add_argument("--output_dir", type=str, default="./neuron_activation_analysis", help="Directory to save the analysis plots.")
    parser.add_argument("--task_type", type=str, required=True, help="The 'type' of text to filter from the dataset (e.g., 'math', 'en').")
    parser.add_argument("--neuron_type", type=str, required=True, help="The type of the neurons to analyze (e.g., 'en', 'ko').")
    parser.add_argument("--module", type=str, default="all", choices=['all', 'mlp', 'q', 'k', 'v'], help="Module to analyze. 'all' runs all modules.")
    parser.add_argument("--top_n_texts", type=int, default=5, help="Number of top activating texts to show per neuron.")

    args = parser.parse_args()

    # --- Korean Font Setup ---
    if 'kor' in args.dataset_path:
        try:
            font_path = fm.findfont(fm.FontProperties(family='NanumGothic'))
            plt.rcParams['font.family'] = 'NanumGothic'
            print(f"NanumGothic font found at: {font_path}")
        except ValueError:
            print("Warning: 'NanumGothic' font not found. Korean text may not display correctly.")
            print("Please install a Korean font (e.g., run 'sudo apt-get install -y fonts-nanum*')")
            print("and clear matplotlib cache ('rm -rf ~/.cache/matplotlib/*').")
        plt.rcParams['axes.unicode_minus'] = False
    # ---

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {args.model_name}")
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    model = HookedTransformer.from_pretrained(args.model_name, device=device, n_devices=n_devices)
    model.eval()

    dataset = load_dataset(args.dataset_path, args.task_type)
    
    model_name_formatted = args.model_name.replace("/", "_")
    task_formatted = args.task_type.replace(" ", "_")
    neuron_formatted = args.neuron_type.replace(" ", "_")
    analysis_desc = f"analyzing_{neuron_formatted}_neurons_on_{task_formatted}_text"
    output_dir_final = os.path.join(args.output_dir, model_name_formatted, analysis_desc)
    os.makedirs(output_dir_final, exist_ok=True)

    modules_to_run = ['mlp', 'q', 'k', 'v'] if args.module == 'all' else [args.module]
    
    top_neurons_base_path = os.path.join(args.analysis_results_dir, 'top_1_percent_neurons', model_name_formatted)

    all_results, source_paths = analyze_activations_single_pass(
        model, dataset, modules_to_run, top_neurons_base_path, 
        model_name_formatted, args.top_n_texts, args.neuron_type
    )

    if not all_results:
        print("Analysis finished with no results to plot.")
        return

    for module, results in all_results.items():
        if results:
            plot_module_results(
                results, module, args.top_n_texts, output_dir_final, 
                source_paths[module], args.task_type, args.neuron_type
            )

if __name__ == "__main__":
    main()