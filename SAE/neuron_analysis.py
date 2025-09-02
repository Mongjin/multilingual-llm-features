
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
import os
import glob

torch.set_grad_enabled(False)

def analyze_and_save_neuron_indices(args):
    """
    저장된 평균 활성화 값을 불러와, 활성화 차이 점수를 계산하고 엔트로피로 필터링하여
    언어 특정적 뉴런 정보를 저장합니다.
    """
    model_name_safe = args.model_name.replace("/", "_")
    input_dir = os.path.join(args.input_dir, model_name_safe)
    output_dir = os.path.join(args.output_dir, model_name_safe)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading average activations from: {input_dir}")
    print(f"Sorted neuron indices will be saved to: {output_dir}")

    activation_files = sorted(glob.glob(os.path.join(input_dir, "layer_*_avg_activations.pth")))

    if not activation_files:
        print(f"Error: No activation files found in {input_dir}. Please run find_language_neurons.py first.")
        return

    for file_path in tqdm(activation_files, desc="Analyzing Layers"):
        data = torch.load(file_path)
        languages = data['languages']
        layer = int(os.path.basename(file_path).split('_')[1])

        # Process MLP activations
        if 'avg_activations_mlp' in data:
            avg_activations_mlp = data['avg_activations_mlp'] # [d_mlp, num_languages]
            
            # 1. 엔트로피 계산
            probs_mlp = F.relu(avg_activations_mlp.T) # [langs, d_mlp]
            probs_mlp = probs_mlp / probs_mlp.sum(dim=0, keepdim=True).clamp_min(1e-8)
            norm_probs_mlp = probs_mlp.clamp_min(1e-12)
            entropy_mlp = -torch.sum(norm_probs_mlp * torch.log(norm_probs_mlp), dim=0)
            
            low_entropy_threshold = torch.quantile(entropy_mlp, args.entropy_quantile)
            low_entropy_indices = (entropy_mlp <= low_entropy_threshold).nonzero(as_tuple=True)[0]

            # 2. 활성화 차이 점수 계산
            all_lang_indices = list(range(len(languages)))
            act_diff_scores_mlp = torch.zeros(len(languages), avg_activations_mlp.shape[0]) # [langs, d_mlp]
            for i, lang in enumerate(languages):
                other_indices = all_lang_indices[:i] + all_lang_indices[i+1:]
                act_diff_scores_mlp[i] = avg_activations_mlp[:, i] - avg_activations_mlp[:, other_indices].mean(dim=1)

            # 3. 엔트로피 필터링 후 정렬
            sorted_scores_mlp, sorted_indices_mlp = torch.sort(act_diff_scores_mlp, dim=1, descending=True)
            
            filtered_indices_list = []
            filtered_scores_list = []
            low_entropy_set = set(low_entropy_indices.tolist())
            for i in range(len(languages)):
                is_low_entropy = torch.tensor([idx.item() in low_entropy_set for idx in sorted_indices_mlp[i]], dtype=torch.bool)
                
                filtered_indices = sorted_indices_mlp[i][is_low_entropy]
                filtered_scores = sorted_scores_mlp[i][is_low_entropy]
                
                filtered_indices_list.append(filtered_indices)
                filtered_scores_list.append(filtered_scores)

            # 4. MLP 결과 저장
            save_path_mlp = os.path.join(output_dir, f"layer_{layer}_top_neurons_mlp.pth")
            data_to_save_mlp = {
                'filtered_indices': {lang: idx.cpu() for lang, idx in zip(languages, filtered_indices_list)},
                'filtered_scores': {lang: score.cpu() for lang, score in zip(languages, filtered_scores_list)},
                'all_scores': act_diff_scores_mlp.cpu(),
                'low_entropy_indices': low_entropy_indices.cpu(),
                'languages': languages
            }
            torch.save(data_to_save_mlp, save_path_mlp)

        # Process Attention activations
        attn_results = {}
        for key in ['avg_activations_q', 'avg_activations_k', 'avg_activations_v']:
            if key in data:
                avg_activations_attn = data[key] # [n_heads, d_head, num_languages]
                avg_activations_attn_flat = avg_activations_attn.permute(2, 0, 1).reshape(len(languages), -1).T

                # 1. 엔트로피 계산
                probs_attn = F.relu(avg_activations_attn_flat.T)
                probs_attn = probs_attn / probs_attn.sum(dim=0, keepdim=True).clamp_min(1e-8)
                norm_probs_attn = probs_attn.clamp_min(1e-12)
                entropy_attn = -torch.sum(norm_probs_attn * torch.log(norm_probs_attn), dim=0)

                low_entropy_threshold_attn = torch.quantile(entropy_attn, args.entropy_quantile)
                low_entropy_indices_attn = (entropy_attn <= low_entropy_threshold_attn).nonzero(as_tuple=True)[0]

                # 2. 활성화 차이 점수 계산
                all_lang_indices = list(range(len(languages)))
                act_diff_scores_attn = torch.zeros(len(languages), avg_activations_attn_flat.shape[0])
                for i, lang in enumerate(languages):
                    other_indices = all_lang_indices[:i] + all_lang_indices[i+1:]
                    act_diff_scores_attn[i] = avg_activations_attn_flat[:, i] - avg_activations_attn_flat[:, other_indices].mean(dim=1)

                # 3. 엔트로피 필터링 후 정렬
                sorted_scores_attn, sorted_indices_attn = torch.sort(act_diff_scores_attn, dim=1, descending=True)

                filtered_indices_list_attn = []
                filtered_scores_list_attn = []
                low_entropy_set_attn = set(low_entropy_indices_attn.tolist())
                for i in range(len(languages)):
                    is_low_entropy_attn = torch.tensor([idx.item() in low_entropy_set_attn for idx in sorted_indices_attn[i]], dtype=torch.bool)
                    
                    filtered_indices_attn = sorted_indices_attn[i][is_low_entropy_attn]
                    filtered_scores_attn = sorted_scores_attn[i][is_low_entropy_attn]
                    
                    filtered_indices_list_attn.append(filtered_indices_attn)
                    filtered_scores_list_attn.append(filtered_scores_attn)
                
                attn_results[key] = {
                    'filtered_indices': {lang: idx.cpu() for lang, idx in zip(languages, filtered_indices_list_attn)},
                    'filtered_scores': {lang: score.cpu() for lang, score in zip(languages, filtered_scores_list_attn)},
                    'all_scores': act_diff_scores_attn.cpu(),
                    'low_entropy_indices': low_entropy_indices_attn.cpu(),
                }

        if attn_results:
            save_path_attn = os.path.join(output_dir, f"layer_{layer}_top_neurons_attn.pth")
            attn_results['languages'] = languages
            torch.save(attn_results, save_path_attn)

    print(f"Analysis complete. Results saved in {output_dir}")

    print(f"Analysis complete. Results saved in {output_dir}")

def find_multilingual_neurons(args):
    """
    Identifies and saves indices of multilingual neurons.
    A multilingual neuron is defined as one that shows high activation across all languages.
    """
    model_name_safe = args.model_name.replace("/", "_")
    input_dir = os.path.join(args.input_dir, model_name_safe)
    output_dir = os.path.join(args.output_dir, model_name_safe)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching for multilingual neurons using activations from: {input_dir}")
    print(f"Multilingual neuron indices will be saved to: {output_dir}")

    activation_files = sorted(glob.glob(os.path.join(input_dir, "layer_*_avg_activations.pth")))

    if not activation_files:
        print(f"Error: No activation files found in {input_dir}. Please run find_language_neurons.py first.")
        return

    for file_path in tqdm(activation_files, desc="Analyzing Layers for Multilingual Neurons"):
        data = torch.load(file_path)
        layer = int(os.path.basename(file_path).split('_')[1])

        # Process MLP activations
        if 'avg_activations_mlp' in data:
            avg_activations_mlp = data['avg_activations_mlp']  # [d_mlp, num_languages]
            
            # Multilingual score: minimum activation across all languages
            multilingual_score_mlp = torch.min(avg_activations_mlp, dim=1).values
            
            sorted_scores_mlp, sorted_indices_mlp = torch.sort(multilingual_score_mlp, descending=True)

            save_path_mlp = os.path.join(output_dir, f"layer_{layer}_multilingual_neurons_mlp.pth")
            data_to_save_mlp = {
                'sorted_indices': sorted_indices_mlp.cpu(),
                'sorted_scores': sorted_scores_mlp.cpu(),
                'languages': data['languages']
            }
            torch.save(data_to_save_mlp, save_path_mlp)

        # Process Attention activations
        attn_results = {}
        for key in ['avg_activations_q', 'avg_activations_k', 'avg_activations_v']:
            if key in data:
                avg_activations_attn = data[key]  # [n_heads, d_head, num_languages]
                avg_activations_attn_flat = avg_activations_attn.permute(2, 0, 1).reshape(len(data['languages']), -1).T

                multilingual_score_attn = torch.min(avg_activations_attn_flat, dim=1).values
                sorted_scores_attn, sorted_indices_attn = torch.sort(multilingual_score_attn, descending=True)
                
                attn_results[key] = {
                    'sorted_indices': sorted_indices_attn.cpu(),
                    'sorted_scores': sorted_scores_attn.cpu(),
                }
        
        if attn_results:
            save_path_attn = os.path.join(output_dir, f"layer_{layer}_multilingual_neurons_attn.pth")
            attn_results['languages'] = data['languages']
            torch.save(attn_results, save_path_attn)

    print(f"Multilingual neuron analysis complete. Results saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-computed activations to find language-specific or multilingual neurons.")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model name for finding the correct subdirectory.")
    parser.add_argument("--input_dir", type=str, default="./neuron_avg_activations_math", help="Directory where average activation files are stored.")
    parser.add_argument("--output_dir", type=str, default="./neuron_analysis_results_math", help="Directory to save the final analysis results.")
    parser.add_argument("--entropy_quantile", type=float, default=0.25, help="Quantile for low-entropy filtering for language-specific neurons.")
    parser.add_argument("--analysis_type", type=str, default="specific", choices=["specific", "multilingual"], help="Type of analysis to perform.")

    args = parser.parse_args()

    if args.analysis_type == "specific":
        analyze_and_save_neuron_indices(args)
    elif args.analysis_type == "multilingual":
        find_multilingual_neurons(args)

if __name__ == "__main__":
    main()
