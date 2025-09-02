import torch
import argparse
import pandas as pd
from tqdm import tqdm
import os
import fnmatch

# Suppress gradient calculations for inference
torch.set_grad_enabled(False)

class ActivationFinder:
    """
    Base class for finding and saving average neuron activations.
    """
    def __init__(self, model_name, dataset_path, output_dir):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.languages = None
        self.lang_to_idx = None

    def load_dataset(self):
        print(f"Loading dataset from: {self.dataset_path}")
        dataset = pd.read_json(self.dataset_path, lines=True)
        self.languages = dataset['lan'].unique().tolist()
        self.lang_to_idx = {lang: i for i, lang in enumerate(self.languages)}
        print(f"Found languages: {self.languages}")
        return dataset

    def run(self):
        raise NotImplementedError

class HookedTransformerActivationFinder(ActivationFinder):
    """
    Activation finder using the HookedTransformer library.
    """
    def __init__(self, model_name, dataset_path, output_dir):
        super().__init__(model_name, dataset_path, output_dir)
        from transformer_lens import HookedTransformer
        print(f"Loading model with HookedTransformer: {self.model_name}")
        n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            n_devices=n_devices,
            device=self.device,
            fold_ln=True, center_unembed=True, center_writing_weights=True
        )
        print(f"Model loaded on {self.model.cfg.n_devices} devices.")
        self.model.eval()

    def run(self):
        from transformer_lens import utils
        dataset = self.load_dataset()
        model_save_dir = os.path.join(self.output_dir, self.model_name.replace("/", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Average activations will be saved to: {model_save_dir}")

        for layer in tqdm(range(self.model.cfg.n_layers), desc="Processing Layers"):
            mlp_hook_point = utils.get_act_name("post", layer)
            attn_hook_point_q = f"blocks.{layer}.attn.hook_q"
            attn_hook_point_k = f"blocks.{layer}.attn.hook_k"
            attn_hook_point_v = f"blocks.{layer}.attn.hook_v"
            
            d_mlp = self.model.cfg.d_mlp
            n_q_heads = self.model.cfg.n_heads
            n_kv_heads = self.model.cfg.n_key_value_heads
            d_head = self.model.cfg.d_head

            total_activations_mlp = torch.zeros((d_mlp, len(self.languages)), device=self.model.embed.W_E.device)
            total_activations_q = torch.zeros((n_q_heads, d_head, len(self.languages)), device=self.model.embed.W_E.device)
            total_activations_k = torch.zeros((n_kv_heads, d_head, len(self.languages)), device=self.model.embed.W_E.device)
            total_activations_v = torch.zeros((n_kv_heads, d_head, len(self.languages)), device=self.model.embed.W_E.device)
            token_counts = torch.zeros(len(self.languages), device=self.model.embed.W_E.device)

            for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Layer {layer} Data", leave=False):
                prompt, lang = row['text'], row['lan']
                if not prompt: continue
                tokens = self.model.to_tokens(prompt, truncate=True)
                try:
                    _, cache = self.model.run_with_cache(tokens, names_filter=[mlp_hook_point, attn_hook_point_q, attn_hook_point_k, attn_hook_point_v])
                    
                    mlp_activations = cache[mlp_hook_point][0]
                    total_activations_mlp[:, self.lang_to_idx[lang]] += mlp_activations.sum(dim=0)

                    q_activations = cache[attn_hook_point_q][0]
                    k_activations = cache[attn_hook_point_k][0]
                    v_activations = cache[attn_hook_point_v][0]
                    
                    total_activations_q[:, :, self.lang_to_idx[lang]] += q_activations.sum(dim=0)
                    total_activations_k[:, :, self.lang_to_idx[lang]] += k_activations.sum(dim=0)
                    total_activations_v[:, :, self.lang_to_idx[lang]] += v_activations.sum(dim=0)

                    token_counts[self.lang_to_idx[lang]] += mlp_activations.shape[0]
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
                'languages': self.languages
            }
            torch.save(data_to_save, save_path)
            print(f"Saved average activations for layer {layer} to {save_path}")

class PytorchHookActivationFinder(ActivationFinder):
    """
    Activation finder using PyTorch hooks for generic transformer models.
    """
    def __init__(self, model_name, dataset_path, output_dir):
        super().__init__(model_name, dataset_path, output_dir)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model with HuggingFace Transformers: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        self.activations = {}
        self.hooks = []

    def get_activation(self, name):
        """Hook function to save activation."""
        def hook(model, input, output):
            # For attention layers, output can be a tuple (activation, weights)
            self.activations[name] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def find_modules(self, model, pattern):
        """Find modules in the model matching a pattern."""
        return [name for name, module in model.named_modules() if fnmatch.fnmatch(name, pattern)]

    def add_hooks(self, layer_idx):
        """Add hooks to the model for a specific layer."""
        # These patterns are educated guesses for Llama-like models.
        # They might need to be adjusted for different architectures.
        mlp_pattern = f"model.layers.{layer_idx}.mlp.up_proj"
        q_pattern = f"model.layers.{layer_idx}.self_attn.q_proj"
        k_pattern = f"model.layers.{layer_idx}.self_attn.k_proj"
        v_pattern = f"model.layers.{layer_idx}.self_attn.v_proj"

        # Find the actual module names
        mlp_modules = self.find_modules(self.model, mlp_pattern)
        q_modules = self.find_modules(self.model, q_pattern)
        k_modules = self.find_modules(self.model, k_pattern)
        v_modules = self.find_modules(self.model, v_pattern)

        # Register the hooks
        if mlp_modules:
            self.hooks.append(self.model.get_submodule(mlp_modules[0]).register_forward_hook(self.get_activation('mlp')))
        if q_modules:
            self.hooks.append(self.model.get_submodule(q_modules[0]).register_forward_hook(self.get_activation('q')))
        if k_modules:
            self.hooks.append(self.model.get_submodule(k_modules[0]).register_forward_hook(self.get_activation('k')))
        if v_modules:
            self.hooks.append(self.model.get_submodule(v_modules[0]).register_forward_hook(self.get_activation('v')))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def run(self):
        dataset = self.load_dataset()
        model_save_dir = os.path.join(self.output_dir, self.model_name.replace("/", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Average activations will be saved to: {model_save_dir}")

        n_layers = self.model.config.num_hidden_layers
        for layer in tqdm(range(n_layers), desc="Processing Layers"):
            self.add_hooks(layer)
            
            # Get model dimensions
            d_mlp = self.model.config.intermediate_size
            n_q_heads = self.model.config.num_attention_heads
            n_kv_heads = self.model.config.num_key_value_heads if hasattr(self.model.config, 'num_key_value_heads') else n_q_heads
            d_head = self.model.config.hidden_size // n_q_heads
            
            # Use the device of the first parameter as the home device
            home_device = next(self.model.parameters()).device

            total_activations_mlp = torch.zeros((d_mlp, len(self.languages)), device=home_device)
            total_activations_q = torch.zeros((n_q_heads, d_head, len(self.languages)), device=home_device)
            total_activations_k = torch.zeros((n_kv_heads, d_head, len(self.languages)), device=home_device)
            total_activations_v = torch.zeros((n_kv_heads, d_head, len(self.languages)), device=home_device)
            token_counts = torch.zeros(len(self.languages), device=home_device)

            for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Layer {layer} Data", leave=False):
                prompt, lang = row['text'], row['lan']
                if not prompt: continue
                
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                
                try:
                    self.activations.clear()
                    _ = self.model(**inputs)

                    # Move activations to the home device for accumulation
                    mlp_activations = self.activations.get('mlp')[0].to(home_device)
                    q_activations = self.activations.get('q')[0].to(home_device)
                    k_activations = self.activations.get('k')[0].to(home_device)
                    v_activations = self.activations.get('v')[0].to(home_device)

                    # Reshape attention heads
                    q_activations = q_activations.view(q_activations.shape[0], n_q_heads, d_head)
                    k_activations = k_activations.view(k_activations.shape[0], n_kv_heads, d_head)
                    v_activations = v_activations.view(v_activations.shape[0], n_kv_heads, d_head)

                    total_activations_mlp[:, self.lang_to_idx[lang]] += mlp_activations.sum(dim=0)
                    total_activations_q[:, :, self.lang_to_idx[lang]] += q_activations.sum(dim=0)
                    total_activations_k[:, :, self.lang_to_idx[lang]] += k_activations.sum(dim=0)
                    total_activations_v[:, :, self.lang_to_idx[lang]] += v_activations.sum(dim=0)

                    token_counts[self.lang_to_idx[lang]] += mlp_activations.shape[0]

                except Exception as e:
                    print(f"Skipping a prompt due to error: {e}")
                    continue
            
            self.remove_hooks()

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
                'languages': self.languages
            }
            torch.save(data_to_save, save_path)
            print(f"Saved average activations for layer {layer} to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute and save per-language average neuron activations.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="Model to analyze.")
    parser.add_argument("--dataset_path", type=str, default="./data/multilingual_data.jsonl", help="Path to the multilingual dataset.")
    parser.add_argument("--output_dir", type=str, default="./neuron_avg_activations_math", help="Directory to save the average activation results.")
    args = parser.parse_args()

    try:
        # Try to load with HookedTransformer to check for support
        from transformer_lens import HookedTransformer
        HookedTransformer.from_pretrained(args.model_name, n_devices=1)
        print("Model is supported by HookedTransformer. Using HookedTransformerActivationFinder.")
        finder = HookedTransformerActivationFinder(args.model_name, args.dataset_path, args.output_dir)
    except Exception as e:
        print(f"Model not supported by HookedTransformer (error: {e}). Falling back to PytorchHookActivationFinder.")
        finder = PytorchHookActivationFinder(args.model_name, args.dataset_path, args.output_dir)

    finder.run()

if __name__ == "__main__":
    main()
