import torch
import argparse
import pandas as pd
from transformer_lens import HookedTransformer, utils
from tqdm import tqdm
import os

torch.set_grad_enabled(False)

def compute_and_save_avg_activations(args):
    """
    모델의 모든 레이어를 순회하며 언어별 평균 뉴런 활성화 값을 계산하고 저장합니다.
    """
    print(f"Loading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=device,
        fold_ln=True, center_unembed=True, center_writing_weights=True
    )
    model.eval()

    print(f"Loading dataset from: {args.dataset_path}")
    dataset = pd.read_json(args.dataset_path, lines=True)
    languages = dataset['lan'].unique().tolist()
    lang_to_idx = {lang: i for i, lang in enumerate(languages)}
    print(f"Found languages: {languages}")

    model_save_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Average activations will be saved to: {model_save_dir}")

    for layer in tqdm(range(model.cfg.n_layers), desc="Processing Layers"):
        mlp_hook_point = utils.get_act_name("post", layer)
        attn_hook_point_q = f"blocks.{layer}.attn.hook_q"
        attn_hook_point_k = f"blocks.{layer}.attn.hook_k"
        attn_hook_point_v = f"blocks.{layer}.attn.hook_v"
        
        d_mlp = model.cfg.d_mlp
        n_heads = model.cfg.n_heads
        d_head = model.cfg.d_head

        total_activations_mlp = torch.zeros((d_mlp, len(languages)), device=device)
        total_activations_q = torch.zeros((n_heads, d_head, len(languages)), device=device)
        total_activations_k = torch.zeros((n_heads, d_head, len(languages)), device=device)
        total_activations_v = torch.zeros((n_heads, d_head, len(languages)), device=device)
        token_counts = torch.zeros(len(languages), device=device)

        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Layer {layer} Data", leave=False):
            prompt, lang = row['text'], row['lan']
            if not prompt: continue
            tokens = model.to_tokens(prompt, truncate=True)
            try:
                _, cache = model.run_with_cache(tokens, names_filter=[mlp_hook_point, attn_hook_point_q, attn_hook_point_k, attn_hook_point_v])
                
                # MLP activations
                mlp_activations = cache[mlp_hook_point][0]
                total_activations_mlp[:, lang_to_idx[lang]] += mlp_activations.sum(dim=0)

                # Attention activations
                q_activations = cache[attn_hook_point_q][0]
                k_activations = cache[attn_hook_point_k][0]
                v_activations = cache[attn_hook_point_v][0]
                
                total_activations_q[:, :, lang_to_idx[lang]] += q_activations.sum(dim=0)
                total_activations_k[:, :, lang_to_idx[lang]] += k_activations.sum(dim=0)
                total_activations_v[:, :, lang_to_idx[lang]] += v_activations.sum(dim=0)

                token_counts[lang_to_idx[lang]] += mlp_activations.shape[0]
            except Exception as e:
                print(f"Skipping a prompt due to error: {e}")
                continue
        
        avg_activations_mlp = total_activations_mlp / (token_counts.unsqueeze(0) + 1e-8)
        avg_activations_q = total_activations_q / (token_counts.view(1, 1, -1) + 1e-8)
        avg_activations_k = total_activations_k / (token_counts.view(1, 1, -1) + 1e-8)
        avg_activations_v = total_activations_v / (token_counts.view(1, 1, -1) + 1e-8)

        save_path = os.path.join(model_save_dir, f"layer_{layer}_avg_activations.pth")
        data_to_save = {
            'avg_activations_mlp': avg_activations_mlp.cpu(),
            'avg_activations_q': avg_activations_q.cpu(),
            'avg_activations_k': avg_activations_k.cpu(),
            'avg_activations_v': avg_activations_v.cpu(),
            'languages': languages
        }
        torch.save(data_to_save, save_path)
        print(f"Saved average activations for layer {layer} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute and save per-language average neuron activations.")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model to analyze.")
    parser.add_argument("--dataset_path", type=str, default="./data/multilingual_data.jsonl", help="Path to the multilingual dataset.")
    parser.add_argument("--output_dir", type=str, default="./neuron_avg_activations", help="Directory to save the average activation results.")

    args = parser.parse_args()
    compute_and_save_avg_activations(args)

if __name__ == "__main__":
    main()