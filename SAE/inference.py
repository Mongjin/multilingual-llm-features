import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functools import partial
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM,  AutoTokenizer
import matplotlib.pyplot as plt
from utils import load_args, load_sae
from tqdm import tqdm



os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


torch.set_grad_enabled(False)  # avoid blowing up mem


class Chat_Model():
    def __init__(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', torch_dtype="auto",)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if  'Meta' in path:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.hooks = []

    
    # Ablate corresponding language features 
    def add_hook_to_change_activation_ablation(self, target_layer, start_idx=0, topk_feature_num=5, ori_lan=4):
        file_dir = f'./sae_acts/{args.model}/layer_{target_layer}/'
        top_index_per_lan = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'), weights_only=True)
        top_index_per_lan = top_index_per_lan[:, start_idx:start_idx+topk_feature_num]
        sae = load_sae(target_layer, args)
        ori_lan_idx = top_index_per_lan[ori_lan]
        if 'Llama' in args.model:
            ori_feature_direction = sae.decoder.weight[:, ori_lan_idx].clone()
        else:
            ori_feature_direction = sae.W_dec.T[:, ori_lan_idx]
        norm = torch.norm(ori_feature_direction, dim=0)**2
        ori_feature_direction = ori_feature_direction / norm


        def change_activation_hook(module, input, output):
            act = output[0]
            if 'Llama' in args.model:
                sae_acts = act.to(torch.bfloat16) @ sae.decoder.weight
            else:
                sae_acts = act.to(torch.float32) @ sae.W_dec.T
            coefficient = sae_acts[0, :, ori_lan_idx].to(act.device)
            act = (act-coefficient@((ori_feature_direction).T)).to(act.dtype)

            return (act, output[1])


        handle = self.model.model.layers[target_layer].register_forward_hook(change_activation_hook)
        self.hooks.append(handle)
    

    def compute_ce_loss(self, data, lan, exclude_lan=True):
        if exclude_lan:
            data = data[data['lan'] != lan]
        else:
            data = data[data['lan'] == lan]
        data = data['text'].to_list()
        neg_log_likelihood = []
        for d in tqdm(data):
            prompts_inputs = self.tokenizer(d, return_tensors="pt").to(self.model.device)
            input_ids = prompts_inputs.input_ids
            target_ids = input_ids.clone()  
            target_ids[:, 0] = -100
            outputs = self.model(input_ids, labels=target_ids)  
            neg_log_likelihood.append(outputs.loss.item()) 
        return neg_log_likelihood


    def add_hook_to_show_activation(self, topk_feature_num=1, ori_lan=-1, new_lan=0):
        # ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
        def show_activation_hook(module, args, kwargs, ori_lan_idx, sae, layer, model_name):
            if 'Llama' in model_name:
                sae_acts = sae.encode(kwargs[0].clone().to(torch.bfloat16))
            else:
                sae_acts = sae.encode(kwargs[0].to(torch.float32))
                # sae_acts = kwargs[0].to(torch.float32) @ sae.W_enc + sae.b_enc
            target = sae_acts[:, :, ori_lan_idx]
            if kwargs[0].shape[1] == 1:
                self.latent_activation[layer] = torch.concat((self.latent_activation[layer], target.cpu()), dim=-1)
            else:
                self.latent_activation.append(target.cpu())
            return
        self.latent_activation = []
        for target_layer in range(self.model.config.num_hidden_layers):
            file_dir = f'./sae_acts/{args.model}/layer_{target_layer}/'
            top_index_per_lan = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'), weights_only=True)
            top_index_per_lan = top_index_per_lan[:, :topk_feature_num]
            sae = load_sae(target_layer, args)
            ori_lan_idx = top_index_per_lan[ori_lan].item()
            handle = self.model.model.layers[target_layer].register_forward_hook(partial(show_activation_hook, ori_lan_idx=ori_lan_idx, sae=sae, layer=target_layer, model_name=args.model))
            self.hooks.append(handle)


    def remove_all_hook(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


    

# Ablate language-specific features and calculate CE loss on head(500) samples for each language corpus
def change_activation_print_ce_corpus_gen(args):
    my_model = Chat_Model(args.model_path)
    multilingual_data = pd.read_json('./data/multilingual_data_test.jsonl', lines=True)
    multilingual_data = multilingual_data.groupby('lan').head(500)
    lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
    print(f'original')
    lan = lan_list[args.target_lan]
    save_dir = f'./plot/line_chart_ce_loss/{args.model}/{lan}'
    os.makedirs(save_dir, exist_ok=True)
    ori_ce_loss = my_model.compute_ce_loss(multilingual_data, 'none', exclude_lan=True)
    sae_ce_loss_all_layer = []
    for layer in tqdm(range(my_model.model.config.num_hidden_layers-(args.modified_layer_num-1))):
        print(f'layer:{layer}')
        args_dict = {'start_idx': args.start_idx, "topk_feature_num": args.topk_feature_num, "ori_lan": lan_list.index(lan)}
        for i in range(args.modified_layer_num):
            my_model.add_hook_to_change_activation_ablation(layer+i, **args_dict)
        ce_loss = my_model.compute_ce_loss(multilingual_data, 'none', exclude_lan=True)
        sae_ce_loss_all_layer.append(ce_loss)
        my_model.remove_all_hook()
    ori_ce_loss = np.array(ori_ce_loss)
    sae_ce_loss_all_layer = np.array(sae_ce_loss_all_layer)
    np.save(os.path.join(save_dir, 'ori_ce_loss.npy'), ori_ce_loss)
    np.save(os.path.join(save_dir, f'sae_ce_loss_all_layer_{args.start_idx}_{args.topk_feature_num}.npy'), sae_ce_loss_all_layer)


# After code switching, measure the increase/elevation in features corresponding to the context language
def code_switch_analysis(args):
    lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh']
    lan_dict = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese', 'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic'}

    # my_model = Chat_Model(args.model_path)
    for target_lan in lan_list:
        my_model = Chat_Model(args.model_path)
        save_dir = f'./plot/line_chart_code_switch/{args.model}/{target_lan}'
        os.makedirs(save_dir, exist_ok=True)
        data = pd.read_json('./data/forced_code_switch.jsonl', lines=True)

        args_dict = {"topk_feature_num": 1, "ori_lan": lan_list.index(target_lan)}
        # my_model.add_hook_to_change_activation(6,**args_dict)

        data = data[data['ori_lan'] == target_lan]
        my_model.add_hook_to_show_activation(**args_dict)
        results = {lan: [] for lan in lan_list}
        results_without_sentence = {lan: [] for lan in lan_list}
        for idx, d in data.iterrows():

            prompts_inputs = my_model.tokenizer(d['sentence'], return_tensors="pt").to(my_model.model.device)
            ori_prompts_inputs = my_model.tokenizer(d['ori_sentence'], return_tensors="pt")
            my_model.model(**prompts_inputs)  # 通过模型计算输出和损失
            latent_activations = torch.concat(my_model.latent_activation)
            latent_activations = latent_activations[:, ori_prompts_inputs.input_ids.shape[1]:]
            my_model.latent_activation = []
            # print(latent_activations)
            # print(d['ori_lan'], d['target_lan'])
            results[d['target_lan']].append(latent_activations)

            my_model.model(torch.concat((prompts_inputs.input_ids[:, :1], prompts_inputs.input_ids[:, ori_prompts_inputs.input_ids.shape[1]:]), dim=-1))  # 通过模型计算输出和损失
            latent_activations = torch.concat(my_model.latent_activation)
            latent_activations = latent_activations[:, 1:]
            my_model.latent_activation = []
            # if d['target_lan'] in ['fr','es','pt']:
            #     print(latent_activations)
            #     print(d['ori_lan'],d['target_lan'])
            results_without_sentence[d['target_lan']].append(latent_activations)
        results_list = []
        results_without_sentence_list = []
        for lan in results.keys():
            results[lan] = torch.concat(results[lan], dim=-1).mean(-1).to(torch.float32)
            results_without_sentence[lan] = torch.concat(results_without_sentence[lan], dim=-1).mean(-1).to(torch.float32)
            if lan != target_lan:
                results_list.append(results[lan])
                results_without_sentence_list.append(results_without_sentence[lan])
        results_all_others = torch.stack(results_list).mean(0)
        results_without_sentence_all_others = torch.stack(results_without_sentence_list).mean(0)
        
        x = list(range(len(results[lan])))

        plt.rcParams.update({
            'font.size': 20,               # Global font size
            'font.weight': 'bold',         # Global font weight (bold)
            'axes.labelweight': 'bold',    # Axis labels
            'axes.titleweight': 'bold',    # Title
        })
        plt.figure(figsize=(10, 4))
        plt.plot(x, results[target_lan], label=f'{lan_dict[target_lan]} Prefix + {lan_dict[target_lan]} Noun', linestyle='-', linewidth=2)
        plt.plot(x, results_all_others, label=f'{lan_dict[target_lan]} Prefix + Other Nouns', linestyle='-', linewidth=2)
        # plt.plot(x, results_without_sentence[target_lan], label=f'{target_lan} Without Prefix', linestyle='-')
        plt.plot(x, results_without_sentence_all_others, label=f'Other Nouns', linestyle='-', linewidth=2)
        # plt.plot(x, results[key]/results[target_lan], label=f'{key}', linestyle='-')
       
        plt.title(f'Activation Value for {lan_dict[target_lan]} Feature')
        plt.xlabel('Layer')
        plt.ylabel(f'Activation Value')


        plt.legend()

        # plt.show()
        plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_{args.model}_{target_lan}.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_{args.model}_{target_lan}.png'), format='png', bbox_inches='tight')
        plt.close()
        for key in results.keys():
            plt.figure(figsize=(10, 4))
            plt.plot(x, results[target_lan], label=f'{lan_dict[target_lan]} Prefix + {lan_dict[target_lan]} Noun', linestyle='-', linewidth=2)
            plt.plot(x, results[key], label=f'{lan_dict[target_lan]} Prefix + {lan_dict[key]} Noun', linestyle='-', linewidth=2)
            # plt.plot(x, results_without_sentence[target_lan], label=f'{target_lan} Without Prefix', linestyle='-')
            plt.plot(x, results_without_sentence[key], label=f'{lan_dict[key]} Noun', linestyle='-', linewidth=2)
            # plt.plot(x, results[key]/results[target_lan], label=f'{key}', linestyle='-')

            plt.title(f'Activation Value for {lan_dict[target_lan]} Feature')
            plt.xlabel('Layer')
            plt.ylabel(f'Activation Value')


            plt.legend()

            # plt.show()
            plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_{args.model}_{target_lan}_{key}.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_{args.model}_{target_lan}_{key}.png'), format='png', bbox_inches='tight')
            plt.close()
        my_model.remove_all_hook()


# After code-switching, measure the decrease/reduction in features corresponding to the original nouns' language
def code_switch_analysis2(args):
    lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh']
    lan_dict = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese', 'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic'}

    # my_model = Chat_Model(args.model_path)
    # noun是相同语言
    for target_lan in lan_list:
        my_model = Chat_Model(args.model_path)
        # save_dir = f'./plot/line_chart_code_switch_ori_noun_decrease/{args.model}/{target_lan}'
        # os.makedirs(save_dir, exist_ok=True)
        data = pd.read_json('./data/forced_code_switch.jsonl', lines=True)

        args_dict = {"topk_feature_num": 1, "ori_lan": lan_list.index(target_lan)}
        # my_model.add_hook_to_change_activation(6,**args_dict)

        data = data[data['target_lan'] == target_lan]
        my_model.add_hook_to_show_activation(**args_dict)
        results = {lan: [] for lan in lan_list}
        results_without_sentence = {lan: [] for lan in lan_list}
        for idx, d in data.iterrows():

            prompts_inputs = my_model.tokenizer(d['sentence'], return_tensors="pt").to(my_model.model.device)
            ori_prompts_inputs = my_model.tokenizer(d['ori_sentence'], return_tensors="pt")
            my_model.model(**prompts_inputs)  
            latent_activations = torch.concat(my_model.latent_activation)
            latent_activations = latent_activations[:, ori_prompts_inputs.input_ids.shape[1]:]
            my_model.latent_activation = []
            # if d['ori_lan'] in ['fr','es','pt']:
            #     pass
            print(latent_activations)
            print(d['ori_lan'], d['target_lan'])
            results[d['ori_lan']].append(latent_activations)

            my_model.model(torch.concat((prompts_inputs.input_ids[:, :1], prompts_inputs.input_ids[:, ori_prompts_inputs.input_ids.shape[1]:]), dim=-1))  # 通过模型计算输出和损失
            latent_activations = torch.concat(my_model.latent_activation)
            latent_activations = latent_activations[:, 1:]
            my_model.latent_activation = []
            print(latent_activations)
            print(d['ori_lan'], d['target_lan'])
            results_without_sentence[d['ori_lan']].append(latent_activations)
        results_list = []
        results_without_sentence_list = []

        for lan in results.keys():
            results[lan] = torch.concat(results[lan], dim=-1).mean(-1).to(torch.float32)
            results_without_sentence[lan] = torch.concat(results_without_sentence[lan], dim=-1).mean(-1).to(torch.float32)
            if lan != target_lan:
                results_list.append(results[lan])
                results_without_sentence_list.append(results_without_sentence[lan])
        results_all_others = torch.stack(results_list).mean(0)
        results_without_sentence_all_others = torch.stack(results_without_sentence_list).mean(0)

        x = list(range(len(results[lan])))

        plt.rcParams.update({
            'font.size': 20,               # Global font size
            'font.weight': 'bold',         # Global font weight (bold)
            'axes.labelweight': 'bold',    # Axis labels
            'axes.titleweight': 'bold',    # Title
        })
        # plt.figure(figsize=(10, 4))
        # # plt.plot(x, results[target_lan], label=f'{lan_dict[target_lan]} With Prefix', linestyle='-',linewidth=2)
        # plt.plot(x, results_all_others, label=f'Prefix + {lan_dict[target_lan]} Noun', linestyle='-',linewidth=2)
        # # plt.plot(x, results_without_sentence[target_lan], label=f'{target_lan} Without Prefix', linestyle='-')
        # plt.plot(x, results_without_sentence_all_others, label=f'{lan_dict[target_lan]} Noun', linestyle='-',linewidth=2)
        # # plt.plot(x, results[key]/results[target_lan], label=f'{key}', linestyle='-')

        # plt.title(f'Activation Value for {lan_dict[target_lan]} Feature')
        # plt.xlabel('Layer')
        # plt.ylabel(f'Activation Value')


        # plt.legend()


        # # plt.show()
        # plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_ori_noun_decrease_{args.model}_{target_lan}.pdf'), format='pdf', bbox_inches='tight')
        # plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_ori_noun_decrease_{args.model}_{target_lan}.png'), format='png', bbox_inches='tight')
        plt.close()
        for key in results.keys():
            plt.figure(figsize=(10, 4))
            # plt.plot(x, results[target_lan], label=f'{lan_dict[target_lan]} With Prefix', linestyle='-',linewidth=2)
            plt.plot(x, results[key], label=f'{lan_dict[key]} Prefix + {lan_dict[target_lan]} Noun', linestyle='-', linewidth=2)
            # plt.plot(x, results_without_sentence[target_lan], label=f'{target_lan} Without Prefix', linestyle='-')
            plt.plot(x, results_without_sentence[key], label=f'{lan_dict[target_lan]} Noun', linestyle='-', linewidth=2)
            # plt.plot(x, results[key]/results[target_lan], label=f'{key}', linestyle='-')
   
            plt.title(f'Activation Value for {lan_dict[target_lan]} Feature')
            plt.xlabel('Layer')
            plt.ylabel(f'Activation Value')

        
            plt.legend()

   
            # plt.show()
            save_dir = f'./plot/line_chart_code_switch_ori_noun_decrease/{args.model}/{key}'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_ori_noun_decrease_{args.model}_{key}_{target_lan}.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'line_chart_code_switch_ori_noun_decrease_{args.model}_{key}_{target_lan}.png'), format='png', bbox_inches='tight')
            plt.close()
        my_model.remove_all_hook()



def topk_feature_results_cal(args):
    """
    Analyzes the top-K features and generates plots showing the tokens
    that maximally activate them for each language and layer.
    """
    print("Starting Top-K Feature Analysis for Plotting...")
    lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
    top_k = 10  # How many top features to analyze for each language

    try:
        multilingual_data = pd.read_json('./data/multilingual_data_test.jsonl', lines=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except FileNotFoundError:
        print("Error: Prerequisite file not found. Ensure model_path is correct and test data exists.")
        return

    for layer in tqdm(range(args.layer_num), desc="Processing Layers"):
        layer_results = []
        file_dir = f'./sae_acts/{args.model}/layer_{layer}/'

        try:
            all_sae_acts = torch.load(os.path.join(file_dir, 'sae_acts.pth'))
            top_indices_per_lan = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'))
        except FileNotFoundError:
            print(f"Activation or index files not found for layer {layer}. Skipping.")
            continue

        for lan_idx, lan_name in enumerate(lan_list):
            top_features_for_lan = top_indices_per_lan[lan_idx, :top_k]
            lan_data = multilingual_data[multilingual_data['lan'] == lan_name]
            lan_act_indices = [i for i, x in enumerate(multilingual_data['lan']) if x == lan_name]

            if not lan_act_indices:
                continue

            for rank, feature_idx in enumerate(top_features_for_lan):
                max_activation_val = -float('inf')
                best_sentence = ""
                best_token = ""

                for i, row in lan_data.iterrows():
                    try:
                        act_idx_in_full_list = lan_act_indices[lan_data.index.get_loc(i)]
                        sae_acts_for_sentence = all_sae_acts[act_idx_in_full_list]
                    except (IndexError, KeyError):
                        continue

                    activations_for_feature = sae_acts_for_sentence[0, :, feature_idx]
                    max_val_in_sentence, max_idx_in_sentence = torch.max(activations_for_feature, dim=0)

                    if max_val_in_sentence > max_activation_val:
                        max_activation_val = max_val_in_sentence.item()
                        inputs = tokenizer.encode(row['text'], return_tensors="pt")
                        token_id = inputs[0, max_idx_in_sentence.item()]
                        best_token = tokenizer.decode(token_id)

                layer_results.append({
                    "lan": lan_name,
                    "rank": rank + 1,
                    "feature_idx": feature_idx.item(),
                    "activation": max_activation_val,
                    "token": best_token
                })

        # After processing all languages for the layer, plot the results
        if layer_results:
            plot_top_feature_activations(layer_results, layer, args, top_k, lan_list)


def plot_top_feature_activations(results, layer, args, top_k, languages):
    """Helper function to plot the results for a single layer."""
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(20, 12), dpi=120)

    x = np.arange(len(languages))
    width = 0.8 / top_k  # Adjust bar width based on K

    # Create a color map for ranks
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))

    for i in range(top_k):
        rank = i + 1
        # Ensure data for all languages exists for consistent plotting
        rank_data = df[df['rank'] == rank].set_index('lan').reindex(languages)

        bar_positions = x - (width * (top_k - 1) / 2) + (i * width)

        bars = ax.bar(bar_positions, rank_data['activation'].fillna(0), width, label=f'Rank {rank}', color=colors[i])

        # Add token annotations on top of bars
        for j, bar in enumerate(bars):
            token = rank_data['token'].iloc[j]
            if pd.notna(token):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, f' {token}',
                        ha='center', va='bottom', rotation=90, fontsize=10, color='black')

    ax.set_ylabel('Max Activation Value', fontsize=14)
    ax.set_title(f'Model: {args.model} | Layer: {layer} | Top {top_k} Activating Tokens per Language Feature', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([lang.upper() for lang in languages], fontsize=12)
    ax.legend(title='Feature Rank')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    save_dir = f'./plot/top_feature_analysis/{args.model}/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'layer_{layer}_top_features_plot.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    args = load_args()
    # topk_feature_results_cal(args)
    change_activation_print_ce_corpus_gen(args)