
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
import os
import glob

torch.set_grad_enabled(False)

def z_normalize(activations):
    """Z-normalizes the activation tensor per layer."""
    mean = torch.mean(activations)
    std = torch.std(activations)
    return (activations - mean) / (std + 1e-8)

def _calculate_scores_and_indices(activations, languages, entropy_quantile):
    """Helper function to calculate entropy, activation differences, and filter neurons."""
    # 1. 엔트로피 계산 (Entropy Calculation)
    probs = F.relu(activations.T)
    probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-8)
    norm_probs = probs.clamp_min(1e-12)
    entropy = -torch.sum(norm_probs * torch.log(norm_probs), dim=0)
    
    low_entropy_threshold = torch.quantile(entropy, entropy_quantile)
    low_entropy_indices = (entropy <= low_entropy_threshold).nonzero(as_tuple=True)[0]

    # 2. 활성화 차이 점수 계산 (Activation Difference Score Calculation)
    all_lang_indices = list(range(len(languages)))
    act_diff_scores = torch.zeros(len(languages), activations.shape[0]) # [num_languages, num_neurons]
    for i, lang in enumerate(languages):
        other_indices = all_lang_indices[:i] + all_lang_indices[i+1:]
        act_diff_scores[i] = activations[:, i] - activations[:, other_indices].mean(dim=1)

    # 3. 엔트로피 필터링 후 정렬 (Sort after Entropy Filtering)
    sorted_scores, sorted_indices = torch.sort(act_diff_scores, dim=1, descending=True)
    
    filtered_indices_list = []
    filtered_scores_list = []
    low_entropy_set = set(low_entropy_indices.tolist())
    for i in range(len(languages)):
        is_low_entropy = torch.tensor([idx.item() in low_entropy_set for idx in sorted_indices[i]], dtype=torch.bool)
        
        filtered_indices = sorted_indices[i][is_low_entropy]
        filtered_scores = sorted_scores[i][is_low_entropy]
        
        filtered_indices_list.append(filtered_indices)
        filtered_scores_list.append(filtered_scores)

    results = {
        'filtered_indices': {lang: idx.cpu() for lang, idx in zip(languages, filtered_indices_list)},
        'filtered_scores': {lang: score.cpu() for lang, score in zip(languages, filtered_scores_list)},
        'all_scores': act_diff_scores.cpu(),
        'low_entropy_indices': low_entropy_indices.cpu(),
    }
    return results


def analyze_and_save_neuron_indices(args):
    """
    저장된 평균 활성화 값을 불러와, 활성화 차이 점수를 계산하고 엔트로피로 필터링하여
    언어 특정적 뉴런 정보를 저장합니다. 정규화된 버전과 정규화되지 않은 버전 모두 저장합니다.
    """
    model_name_safe = args.model_name.replace("/", "_")
    input_dir = os.path.join(args.input_dir, model_name_safe)
    
    # Create separate output directories for original and normalized results
    output_dir_orig = os.path.join(args.output_dir, model_name_safe)
    output_dir_norm = os.path.join(args.output_dir + "_normalized", model_name_safe)
    os.makedirs(output_dir_orig, exist_ok=True)
    os.makedirs(output_dir_norm, exist_ok=True)

    print(f"Loading average activations from: {input_dir}")
    print(f"Original sorted neuron indices will be saved to: {output_dir_orig}")
    print(f"Normalized sorted neuron indices will be saved to: {output_dir_norm}")

    activation_files = sorted(glob.glob(os.path.join(input_dir, "layer_*_avg_activations.pth")))

    if not activation_files:
        print(f"Error: No activation files found in {input_dir}. Please run find_language_neurons.py first.")
        return

    for file_path in tqdm(activation_files, desc="Analyzing Layers"):
        data = torch.load(file_path)
        languages = data['languages']
        layer = int(os.path.basename(file_path).split('_')[1])

        # --- Process MLP activations ---
        if 'avg_activations_mlp' in data:
            avg_activations_mlp = data['avg_activations_mlp'] # [d_mlp, num_languages]

            # 1. Original analysis
            results_mlp_orig = _calculate_scores_and_indices(avg_activations_mlp, languages, args.entropy_quantile)
            results_mlp_orig['languages'] = languages
            save_path_mlp_orig = os.path.join(output_dir_orig, f"layer_{layer}_top_neurons_mlp.pth")
            torch.save(results_mlp_orig, save_path_mlp_orig)

            # 2. Z-normalized analysis
            normalized_activations_mlp = z_normalize(avg_activations_mlp)
            results_mlp_norm = _calculate_scores_and_indices(normalized_activations_mlp, languages, args.entropy_quantile)
            results_mlp_norm['languages'] = languages
            save_path_mlp_norm = os.path.join(output_dir_norm, f"layer_{layer}_top_neurons_mlp.pth")
            torch.save(results_mlp_norm, save_path_mlp_norm)

        # --- Process Attention activations ---
        attn_results_orig = {}
        attn_results_norm = {}
        for key in ['avg_activations_q', 'avg_activations_k', 'avg_activations_v']:
            if key in data:
                avg_activations_attn = data[key]
                avg_activations_attn_flat = avg_activations_attn.permute(2, 0, 1).reshape(len(languages), -1).T

                # 1. Original analysis
                results_attn_orig = _calculate_scores_and_indices(avg_activations_attn_flat, languages, args.entropy_quantile)
                attn_results_orig[key] = results_attn_orig

                # 2. Z-normalized analysis
                normalized_activations_attn = z_normalize(avg_activations_attn_flat)
                results_attn_norm = _calculate_scores_and_indices(normalized_activations_attn, languages, args.entropy_quantile)
                attn_results_norm[key] = results_attn_norm

        if attn_results_orig:
            attn_results_orig['languages'] = languages
            save_path_attn_orig = os.path.join(output_dir_orig, f"layer_{layer}_top_neurons_attn.pth")
            torch.save(attn_results_orig, save_path_attn_orig)

        if attn_results_norm:
            attn_results_norm['languages'] = languages
            save_path_attn_norm = os.path.join(output_dir_norm, f"layer_{layer}_top_neurons_attn.pth")
            torch.save(attn_results_norm, save_path_attn_norm)

    print(f"Analysis complete. Original results saved in {output_dir_orig}")
    print(f"Analysis complete. Normalized results saved in {output_dir_norm}")

def find_multilingual_neurons(args):
    """
    Identifies and saves indices of multilingual neurons.
    A multilingual neuron is defined as one that shows high activation across all languages.
    Saves both original and z-normalized results.
    """
    model_name_safe = args.model_name.replace("/", "_")
    input_dir = os.path.join(args.input_dir, model_name_safe)
    
    output_dir_orig = os.path.join(args.output_dir, model_name_safe)
    output_dir_norm = os.path.join(args.output_dir + "_normalized", model_name_safe)
    os.makedirs(output_dir_orig, exist_ok=True)
    os.makedirs(output_dir_norm, exist_ok=True)

    print(f"Searching for multilingual neurons using activations from: {input_dir}")
    print(f"Original multilingual neuron indices will be saved to: {output_dir_orig}")
    print(f"Normalized multilingual neuron indices will be saved to: {output_dir_norm}")

    activation_files = sorted(glob.glob(os.path.join(input_dir, "layer_*_avg_activations.pth")))

    if not activation_files:
        print(f"Error: No activation files found in {input_dir}. Please run find_language_neurons.py first.")
        return

    for file_path in tqdm(activation_files, desc="Analyzing Layers for Multilingual Neurons"):
        data = torch.load(file_path)
        layer = int(os.path.basename(file_path).split('_')[1])

        # Process MLP activations
        if 'avg_activations_mlp' in data:
            avg_activations_mlp = data['avg_activations_mlp'] # [d_mlp, num_languages]
            
            # Original
            multilingual_score_mlp = torch.min(avg_activations_mlp, dim=1).values
            sorted_scores_mlp, sorted_indices_mlp = torch.sort(multilingual_score_mlp, descending=True)
            save_path_mlp_orig = os.path.join(output_dir_orig, f"layer_{layer}_multilingual_neurons_mlp.pth")
            torch.save({
                'sorted_indices': sorted_indices_mlp.cpu(),
                'sorted_scores': sorted_scores_mlp.cpu(),
                'languages': data['languages']
            }, save_path_mlp_orig)

            # Normalized
            normalized_activations_mlp = z_normalize(avg_activations_mlp)
            multilingual_score_mlp_norm = torch.min(normalized_activations_mlp, dim=1).values
            sorted_scores_mlp_norm, sorted_indices_mlp_norm = torch.sort(multilingual_score_mlp_norm, descending=True)
            save_path_mlp_norm = os.path.join(output_dir_norm, f"layer_{layer}_multilingual_neurons_mlp.pth")
            torch.save({
                'sorted_indices': sorted_indices_mlp_norm.cpu(),
                'sorted_scores': sorted_scores_mlp_norm.cpu(),
                'languages': data['languages']
            }, save_path_mlp_norm)

        # Process Attention activations
        attn_results_orig = {}
        attn_results_norm = {}
        for key in ['avg_activations_q', 'avg_activations_k', 'avg_activations_v']:
            if key in data:
                avg_activations_attn = data[key]
                avg_activations_attn_flat = avg_activations_attn.permute(2, 0, 1).reshape(len(data['languages']), -1).T

                # Original
                multilingual_score_attn = torch.min(avg_activations_attn_flat, dim=1).values
                sorted_scores_attn, sorted_indices_attn = torch.sort(multilingual_score_attn, descending=True)
                attn_results_orig[key] = {
                    'sorted_indices': sorted_indices_attn.cpu(),
                    'sorted_scores': sorted_scores_attn.cpu(),
                }

                # Normalized
                normalized_activations_attn = z_normalize(avg_activations_attn_flat)
                multilingual_score_attn_norm = torch.min(normalized_activations_attn, dim=1).values
                sorted_scores_attn_norm, sorted_indices_attn_norm = torch.sort(multilingual_score_attn_norm, descending=True)
                attn_results_norm[key] = {
                    'sorted_indices': sorted_indices_attn_norm.cpu(),
                    'sorted_scores': sorted_scores_attn_norm.cpu(),
                }
        
        if attn_results_orig:
            attn_results_orig['languages'] = data['languages']
            save_path_attn_orig = os.path.join(output_dir_orig, f"layer_{layer}_multilingual_neurons_attn.pth")
            torch.save(attn_results_orig, save_path_attn_orig)
        
        if attn_results_norm:
            attn_results_norm['languages'] = data['languages']
            save_path_attn_norm = os.path.join(output_dir_norm, f"layer_{layer}_multilingual_neurons_attn.pth")
            torch.save(attn_results_norm, save_path_attn_norm)

    print(f"Multilingual neuron analysis complete. Original results saved in {output_dir_orig}")
    print(f"Multilingual neuron analysis complete. Normalized results saved in {output_dir_norm}")


def find_bilingual_neurons(args):
    """
    Identifies and saves indices of bilingual neurons for two specified languages.
    A bilingual neuron is defined as one that shows high activation for both languages.
    Saves both original and z-normalized results.
    """
    if not args.lang1 or not args.lang2:
        print("Error: --lang1 and --lang2 are required for bilingual analysis.")
        return

    model_name_safe = args.model_name.replace("/", "_")
    input_dir = os.path.join(args.input_dir, model_name_safe)
    
    output_dir_orig = os.path.join(args.output_dir, model_name_safe)
    output_dir_norm = os.path.join(args.output_dir + "_normalized", model_name_safe)
    os.makedirs(output_dir_orig, exist_ok=True)
    os.makedirs(output_dir_norm, exist_ok=True)

    print(f"Searching for bilingual neurons for {args.lang1} and {args.lang2} using activations from: {input_dir}")
    print(f"Original bilingual neuron indices will be saved to: {output_dir_orig}")
    print(f"Normalized bilingual neuron indices will be saved to: {output_dir_norm}")

    activation_files = sorted(glob.glob(os.path.join(input_dir, "layer_*_avg_activations.pth")))

    if not activation_files:
        print(f"Error: No activation files found in {input_dir}. Please run find_language_neurons.py first.")
        return

    for file_path in tqdm(activation_files, desc=f"Analyzing Layers for {args.lang1}-{args.lang2} Bilingual Neurons"):
        data = torch.load(file_path)
        languages = data['languages']
        layer = int(os.path.basename(file_path).split('_')[1])

        try:
            lang1_idx = languages.index(args.lang1)
            lang2_idx = languages.index(args.lang2)
        except ValueError as e:
            print(f"Error: One of the languages not found in layer {layer}: {e}")
            continue

        # Process MLP activations
        if 'avg_activations_mlp' in data:
            avg_activations_mlp = data['avg_activations_mlp'] # [d_mlp, num_languages]
            
            # Original
            bilingual_score_mlp = torch.min(avg_activations_mlp[:, [lang1_idx, lang2_idx]], dim=1).values
            sorted_scores_mlp, sorted_indices_mlp = torch.sort(bilingual_score_mlp, descending=True)
            save_path_mlp_orig = os.path.join(output_dir_orig, f"layer_{layer}_bilingual_{args.lang1}_{args.lang2}_neurons_mlp.pth")
            torch.save({
                'sorted_indices': sorted_indices_mlp.cpu(),
                'sorted_scores': sorted_scores_mlp.cpu(),
                'languages': [args.lang1, args.lang2]
            }, save_path_mlp_orig)

            # Normalized
            normalized_activations_mlp = z_normalize(avg_activations_mlp)
            bilingual_score_mlp_norm = torch.min(normalized_activations_mlp[:, [lang1_idx, lang2_idx]], dim=1).values
            sorted_scores_mlp_norm, sorted_indices_mlp_norm = torch.sort(bilingual_score_mlp_norm, descending=True)
            save_path_mlp_norm = os.path.join(output_dir_norm, f"layer_{layer}_bilingual_{args.lang1}_{args.lang2}_neurons_mlp.pth")
            torch.save({
                'sorted_indices': sorted_indices_mlp_norm.cpu(),
                'sorted_scores': sorted_scores_mlp_norm.cpu(),
                'languages': [args.lang1, args.lang2]
            }, save_path_mlp_norm)

        # Process Attention activations
        attn_results_orig = {}
        attn_results_norm = {}
        for key in ['avg_activations_q', 'avg_activations_k', 'avg_activations_v']:
            if key in data:
                avg_activations_attn = data[key]
                avg_activations_attn_flat = avg_activations_attn.permute(2, 0, 1).reshape(len(languages), -1).T

                # Original
                bilingual_score_attn = torch.min(avg_activations_attn_flat[:, [lang1_idx, lang2_idx]], dim=1).values
                sorted_scores_attn, sorted_indices_attn = torch.sort(bilingual_score_attn, descending=True)
                attn_results_orig[key] = {
                    'sorted_indices': sorted_indices_attn.cpu(),
                    'sorted_scores': sorted_scores_attn.cpu(),
                }

                # Normalized
                normalized_activations_attn = z_normalize(avg_activations_attn_flat)
                bilingual_score_attn_norm = torch.min(normalized_activations_attn[:, [lang1_idx, lang2_idx]], dim=1).values
                sorted_scores_attn_norm, sorted_indices_attn_norm = torch.sort(bilingual_score_attn_norm, descending=True)
                attn_results_norm[key] = {
                    'sorted_indices': sorted_indices_attn_norm.cpu(),
                    'sorted_scores': sorted_scores_attn_norm.cpu(),
                }
        
        if attn_results_orig:
            attn_results_orig['languages'] = [args.lang1, args.lang2]
            save_path_attn_orig = os.path.join(output_dir_orig, f"layer_{layer}_bilingual_{args.lang1}_{args.lang2}_neurons_attn.pth")
            torch.save(attn_results_orig, save_path_attn_orig)

        if attn_results_norm:
            attn_results_norm['languages'] = [args.lang1, args.lang2]
            save_path_attn_norm = os.path.join(output_dir_norm, f"layer_{layer}_bilingual_{args.lang1}_{args.lang2}_neurons_attn.pth")
            torch.save(attn_results_norm, save_path_attn_norm)

    print(f"Bilingual neuron analysis complete. Original results saved in {output_dir_orig}")
    print(f"Bilingual neuron analysis complete. Normalized results saved in {output_dir_norm}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-computed activations to find language-specific or multilingual neurons.")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model name for finding the correct subdirectory.")
    parser.add_argument("--input_dir", type=str, default="./neuron_avg_activations_math", help="Directory where average activation files are stored.")
    parser.add_argument("--output_dir", type=str, default="./neuron_analysis_results_math", help="Directory to save the final analysis results.")
    parser.add_argument("--entropy_quantile", type=float, default=0.25, help="Quantile for low-entropy filtering for language-specific neurons.")
    parser.add_argument("--analysis_type", type=str, default="specific", choices=["specific", "multilingual", "bilingual"], help="Type of analysis to perform.")
    parser.add_argument("--lang1", type=str, help="First language for bilingual analysis.")
    parser.add_argument("--lang2", type=str, help="Second language for bilingual analysis.")

    args = parser.parse_args()

    if args.analysis_type == "specific":
        analyze_and_save_neuron_indices(args)
    elif args.analysis_type == "multilingual":
        find_multilingual_neurons(args)
    elif args.analysis_type == "bilingual":
        find_bilingual_neurons(args)

if __name__ == "__main__":
    main()
