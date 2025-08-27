
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
import os
import glob

torch.set_grad_enabled(False)

def normalize_scores(scores):
    """Min-Max 정규화를 통해 점수를 0과 1 사이로 조정합니다."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return torch.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def analyze_and_save_neuron_indices(args):
    """
    저장된 평균 활성화 값을 불러와, 활성화 차이와 엔트로피를 결합한 점수로
    언어 특정적 뉴런의 인덱스를 정렬하여 저장합니다.
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
        avg_activations = data['avg_activations'] # [d_mlp, num_languages]
        languages = data['languages']
        layer = int(os.path.basename(file_path).split('_')[1])

        # 1. 활성화 차이 점수 계산
        all_lang_indices = list(range(len(languages)))
        act_diff_scores = torch.zeros(len(languages), avg_activations.shape[0]) # [langs, d_mlp]
        for i, lang in enumerate(languages):
            other_indices = all_lang_indices[:i] + all_lang_indices[i+1:]
            act_diff_scores[i] = avg_activations[:, i] - avg_activations[:, other_indices].mean(dim=1)

        # 2. 엔트로피 점수 계산
        probs = F.relu(avg_activations.T) # [langs, d_mlp]
        # norm_probs = probs / (probs.sum(dim=0, keepdim=True) + 1e-8)
        probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        norm_probs = probs.clamp_min(1e-12)  # log(0) 방지
        # entropy = -torch.sum(norm_probs * torch.log(norm_probs + 1e-9), dim=0) # [d_mlp]
        entropy = -torch.sum(norm_probs * torch.log(norm_probs), dim=0)
        entropy_scores = -entropy

        # 3. 점수 결합 및 정렬
        norm_act_diff = torch.stack([normalize_scores(s) for s in act_diff_scores])
        norm_entropy = normalize_scores(entropy_scores)

        # Check the scale of normalized scores
        print(f"\nLayer {layer}:")
        print(f"  norm_act_diff - Min: {norm_act_diff.min():.4f}, Max: {norm_act_diff.max():.4f}, Mean: {norm_act_diff.mean():.4f}")
        print(f"  norm_entropy  - Min: {norm_entropy.min():.4f}, Max: {norm_entropy.max():.4f}, Mean: {norm_entropy.mean():.4f}")

        combined_scores = norm_act_diff * norm_entropy.unsqueeze(0)

        sorted_indices = torch.argsort(combined_scores, dim=1, descending=True)

        # 4. 결과 저장
        save_path = os.path.join(output_dir, f"layer_{layer}_top_neurons_combined.pth")
        data_to_save = {
            'sorted_indices': sorted_indices.cpu(),
            'scores': combined_scores.cpu(),
            'languages': languages
        }
        torch.save(data_to_save, save_path)
    
    print(f"Analysis complete. Results saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze pre-computed activations to find language-specific neurons.")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model name for finding the correct subdirectory.")
    parser.add_argument("--input_dir", type=str, default="./neuron_avg_activations", help="Directory where average activation files are stored.")
    parser.add_argument("--output_dir", type=str, default="./neuron_analysis_results", help="Directory to save the final analysis results.")

    args = parser.parse_args()
    analyze_and_save_neuron_indices(args)

if __name__ == "__main__":
    main()
