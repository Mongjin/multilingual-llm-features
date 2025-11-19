import argparse
import json
from transformers import AutoTokenizer
import os

def main():
    parser = argparse.ArgumentParser(description="Compare tokenizers.")
    parser.add_argument("--model_list", nargs="+", required=True, help="List of Hugging Face model paths.")
    parser.add_argument("--custom_tokenizer_path", help="Path to the custom tokenizer.", default='./final_cand_132k_rebalanced-eng55p_kor20p_250828')
    parser.add_argument("--input_data_path", help="Path to the input data file (JSONL format with a 'text' field).", default='./data/multitask_en_kor.jsonl')
    parser.add_argument("--output_path", default="tokenization_comparison.json", help="Path to save the output JSON file.")
    args = parser.parse_args()

    # Load tokenizers, preferring fast tokenizers
    tokenizers = {}
    all_model_paths = args.model_list + [args.custom_tokenizer_path]
    tokenizer_names = args.model_list + ["custom"]

    for name, model_path in zip(tokenizer_names, all_model_paths):
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        except Exception:
            try:
                tokenizers[name] = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                print(f"Warning: Could not load fast tokenizer for {model_path}. Falling back to slow tokenizer.")
            except Exception as e:
                print(f"Error loading tokenizer for {model_path}: {e}")

    # Process data
    lengths = {name: 0 for name in tokenizers.keys()}
    results = {name: [] for name in tokenizers.keys()}
    with open(args.input_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                text_type = data.get("type", "")
                if text_type != 'math':
                    continue
                if not text:
                    continue

                for name, tokenizer in tokenizers.items():
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    
                    decoded_tokens = []
                    for token_id in token_ids:
                        decoded_tokens.append(tokenizer.decode([token_id]))

                    results[name].append({
                        "text": text,
                        "tokens": tokens,
                        "decoded_tokens": decoded_tokens,
                        "length": len(token_ids)
                    })
                    lengths[name] += len(token_ids)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"An error occurred while processing a line: {e}")
    for name in tokenizers.keys():
        lengths[name] = lengths[name] / len(results[name])
    # Save results
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({"average_lengths": lengths}, f, ensure_ascii=False, indent=2)
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Tokenization comparison saved to {args.output_path}")

if __name__ == "__main__":
    main()