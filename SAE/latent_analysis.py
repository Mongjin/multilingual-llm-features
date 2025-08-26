import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


# Sorted indices based on descending values of the monolingual metric (absolute activation differences)
def generate_top_index_magnitude(args):
    for layer in tqdm(range(args.layer_num)):
        # layer=0
        file_dir = f'./sae_acts/{args.model}/layer_{layer}/'
        all_sae_acts = torch.load(os.path.join(file_dir, 'sae_acts.pth'))
    
        all_sae_acts_per_token = []
        for acts in all_sae_acts:
            all_sae_acts_per_token.append(acts[0, 1:, :])
        
        # 데이터셋 정보를 직접 로드하여 언어 목록과 그룹을 동적으로 찾음
        multilingual_data = pd.read_json('./data/multilingual_data.jsonl', lines=True)
        # multilingual_data = pd.read_json('./data/metamath_thinking.jsonl', lines=True)
        # 데이터에 존재하는 고유한 언어 목록을 가져옴
        lan_list = multilingual_data['lan'].unique()
        num_lan = len(lan_list)

        avg_act_per_lan = []
        # 실제 언어 이름을 기준으로 반복
        for lan_name in lan_list:
            # 현재 언어(lan_name)에 해당하는 모든 문장의 인덱스를 찾음
            lan_indices = multilingual_data.index[multilingual_data['lan'] == lan_name].tolist()

            # 해당 인덱스를 사용하여 활성화 값 리스트에서 올바른 활성화 값을 추출
            lan_acts = [all_sae_acts_per_token[i] for i in lan_indices]

            # 추출된 활성화 값들을 하나로 합침
            all_sae_acts_per_token_lan = torch.concat(lan_acts)
            avg_act = all_sae_acts_per_token_lan.mean(dim=-2)
            avg_act_per_lan.append(avg_act)

        avg_act_per_lan = torch.stack(avg_act_per_lan)

        # avg_act_per_lan = []
        # for i in range(num_lan):
        #     all_sae_acts_per_token_lan = torch.concat(all_sae_acts_per_token[100*i:100*(i+1)])
        #     avg_act = all_sae_acts_per_token_lan.mean(dim=-2)
        #     avg_act_per_lan.append(avg_act)
        # avg_act_per_lan = torch.stack(avg_act_per_lan)

        top_index_per_lan = []
        top_ratio_per_lan = []
        for i in range(num_lan):
            avg_act_difference_per_lan=avg_act_per_lan[i]-torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i+1:]], dim=0).mean(dim=0)
            sorted_values, sorted_indices=torch.sort(avg_act_difference_per_lan, descending=True)
            top_ratio_per_lan.append(sorted_values.unsqueeze(0))
            top_index_per_lan.append(sorted_indices.unsqueeze(0))
        top_index_per_lan = torch.concat(top_index_per_lan)
        top_ratio_per_lan = torch.concat(top_ratio_per_lan)
        torch.save(top_index_per_lan, os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'))


if __name__ == "__main__":
    args = load_args()
    generate_top_index_magnitude(args)
