import os
import json
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM,  AutoTokenizer
import numpy as np
import torch
from utils import load_args, load_sae

torch.set_grad_enabled(False)  # avoid blowing up mem

def calculate_entropy(activations, epsilon=1e-9):
    """Calculates the entropy of a feature's activations."""
    activations = torch.clamp(activations, min=0)
    total_activation = activations.sum()
    if total_activation == 0:
        return 0.0
    probs = activations / total_activation
    probs += epsilon
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()

def generate_top_feature_indices(args, entropy_quantile=0.25):
    """
    Generates and saves indices and scores of top language-specific features
    based on both activation magnitude and low entropy.
    """
    for layer in tqdm(range(args.layer_num), desc="Processing Layers"):
        file_dir = f'./sae_acts/{args.model}/layer_{layer}/'
        os.makedirs(file_dir, exist_ok=True)
        
        try:
            all_sae_acts = torch.load(os.path.join(file_dir, 'sae_acts.pth'))
        except FileNotFoundError:
            print(f"SAE activations file not found for layer {layer}. Skipping.")
            continue
        
        # shape of each element in all_sae_acts: (seq_len, feature_dim)
        # Concatenate all token activations across all samples, excluding the initial token
        if 'Llama' in args.model:
            all_sae_acts_per_token = torch.cat([acts[1:, :] for acts in all_sae_acts if acts.shape[0] > 1])
        else:    
            all_sae_acts_per_token = torch.cat([acts[0, 1:, :] for acts in all_sae_acts if acts.shape[1] > 1])

        num_features = all_sae_acts_per_token.shape[-1]
        feature_entropies = torch.zeros(num_features)
        for i in tqdm(range(num_features), desc=f"Calculating Entropy for Layer {layer}", leave=False):
            feature_activations = all_sae_acts_per_token[:, i]
            feature_entropies[i] = calculate_entropy(feature_activations)
        
        low_entropy_threshold = torch.quantile(feature_entropies, entropy_quantile)
        low_entropy_feature_indices = (feature_entropies <= low_entropy_threshold).nonzero(as_tuple=True)[0]
        low_entropy_set = set(low_entropy_feature_indices.tolist())

        multilingual_data = pd.read_json(args.feature_data_path, lines=True)
        lan_list = multilingual_data['lan'].unique()
        num_lan = len(lan_list)

        avg_act_per_lan = []
        if 'Llama' in args.model:
            all_sae_acts_per_token_list = [acts[1:, :] for acts in all_sae_acts if acts.shape[0] > 1]
        else:
            all_sae_acts_per_token_list = [acts[0, 1:, :] for acts in all_sae_acts if acts.shape[1] > 1]
        
        lan_indices_map = {lan_name: [] for lan_name in lan_list}
        for i, lan_name in enumerate(multilingual_data['lan']):
            lan_indices_map[lan_name].append(i)

        for lan_name in lan_list:
            lan_acts = [all_sae_acts_per_token_list[i] for i in lan_indices_map[lan_name]]
            if not lan_acts: continue
            all_sae_acts_per_token_lan = torch.cat(lan_acts)
            avg_act = all_sae_acts_per_token_lan.mean(dim=-2)
            avg_act_per_lan.append(avg_act)
        avg_act_per_lan = torch.stack(avg_act_per_lan)

        all_scores_list = []
        for i in range(num_lan):
            score = avg_act_per_lan[i] - torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i+1:]], dim=0).mean(dim=0)
            all_scores_list.append(score)
        all_scores = torch.stack(all_scores_list)

        # Save the new comprehensive analysis file
        feature_analysis_data = {
            'all_scores': all_scores.cpu(),
            'low_entropy_indices': low_entropy_feature_indices.cpu(),
            'languages': list(lan_list)
        }
        torch.save(feature_analysis_data, os.path.join(file_dir, 'sae_feature_analysis.pth'))
        print(f"Saved full feature analysis data for layer {layer}.")

        top_indices_magnitude_only = []
        top_indices_mag_and_entropy = {}

        for i, lan_name in enumerate(lan_list):
            # Use the already computed scores
            avg_act_difference_per_lan = all_scores[i]
            sorted_values_magnitude, sorted_indices_magnitude = torch.sort(avg_act_difference_per_lan, descending=True)
            
            top_indices_magnitude_only.append(sorted_indices_magnitude.unsqueeze(0))

            is_low_entropy = torch.zeros(num_features, dtype=torch.bool)
            if low_entropy_set:
                 is_low_entropy[list(low_entropy_set)] = True

            entropy_filtered_mask = is_low_entropy[sorted_indices_magnitude]
            
            filtered_indices = sorted_indices_magnitude[entropy_filtered_mask]
            filtered_scores = sorted_values_magnitude[entropy_filtered_mask]
            
            top_indices_mag_and_entropy[lan_name] = (filtered_indices.cpu(), filtered_scores.cpu())

        top_indices_magnitude_only = torch.cat(top_indices_magnitude_only)
        torch.save(top_indices_magnitude_only, os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'))

        torch.save(top_indices_mag_and_entropy, os.path.join(file_dir, 'top_index_per_lan_magnitude_entropy.pth'))
        print(f"Saved magnitude-only and magnitude+entropy filtered (indices, scores) for layer {layer}.")


if __name__ == "__main__":
    args = load_args()
    generate_top_feature_indices(args)
