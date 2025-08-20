# Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders

[![arXiv](https://img.shields.io/badge/arXiv-2505.05111-b31b1b.svg)](https://arxiv.org/abs/2505.05111)
[![Conference](https://img.shields.io/badge/ACL-2025-4b44ce.svg)](https://www.2025.aclweb.org/)

This repository contains the official implementation of "Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders", accepted at ACL 2025.

## Installation

```bash
git clone https://github.com/username/multilingual-llm-features.git
cd multilingual-llm-features
pip install -r requirements.txt
```

### Additional Setup for Llama-3.1-8B

If you plan to work with the Llama-3.1-8B SAE, you'll need to install additional dependencies following the guidelines from [OpenMOSS/Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs):


## Downloading Pre-trained SAEs

To download the pre-trained Sparse Autoencoders (SAEs), simply run:

```bash
python download.py
```

This will automatically download the following SAE models in the `SAE` directory:
- `Llama3_1-8B-Base-LXR-8x`
- `gemma-scope-2b-pt-res`
- `gemma-scope-9b-pt-res`

The directory structure after downloading will be:
```
SAE/
├── Llama3_1-8B-Base-LXR-8x/
├── gemma-scope-2b-pt-res/
└── gemma-scope-9b-pt-res/
```

Note: Make sure you have sufficient disk space and a stable internet connection before running the download script.

If you encounter network issues, you can uncomment the alternative download URLs in `download.py` to use mirror sites for downloading the models.

## Usage

### 1. Finding Language-Specific Features

To identify language-specific features in LLMs, follow these steps (an example for gemma-2-2b):

```bash
# Step 1: Compute latent representations of language features
python latent_computation.py --model "gemma-2-2b" --model_path YOUR_MODEL_PATH 
# For additional arguments, please refer to utils.py

# Step 2: Analyze and extract language-specific features
python latent_analysis.py --model "gemma-2-2b" --model_path YOUR_MODEL_PATH
```

The results will be saved in the `sae_acts` directory with the following structure:
```
sae_acts/
└── gemma-2b/
    └── layer_3/
        ├── top_index_per_lan_magnitude.pth  # Language-specific features ranked by magnitude
        ├── sae_acts.pth                     # Intermediate results
        └── ...
```

Note: `top_index_per_lan_magnitude.pth` has dimensions of 10 × feature_num, where 10 represents different languages in the following order: ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar'] (English, Spanish, French, Japanese, Korean, Portuguese, Thai, Vietnamese, Chinese, Arabic).


### 2. Reproducing Code-Switching Analysis (Section 4)

To reproduce the code-switching analysis results:

1. Execute `code_switch_analysis()` function in `inference.py`
2. Run `code_switch_analysis2()` function in `inference.py` 

The visualization results will be automatically saved in the `plot` directory.

### 3. Feature Ablation Analysis (Section 5)

To reproduce the analysis of language-specific feature ablation:

1. Run `change_activation_print_ce_corpus_gen()` function in `inference.py` with the following parameter combinations:
   - `--start_idx 0 --topk_feature_num 1`
   - `--start_idx 0 --topk_feature_num 2`
   - `--start_idx 1 --topk_feature_num 1`

2. Generate visualizations by running:
   - `change_activation_print_ce_corpus_different_same_feature_diff_lan_all()`
   - `change_activation_print_ce_corpus()`
   functions in `plot.py`

All visualization outputs will be stored in the `plot` directory for further analysis.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{deng2025unveiling,
  title={Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders},
  author={Deng, Boyi and Wan, Yu and Zhang, Yidan and Yang, Baosong and Feng, Fuli},
  journal={arXiv preprint arXiv:2505.05111},
  year={2025}
}
```

## Contact

For any questions or feedback, feel free to reach out to Boyi Deng at dengboyi@mail.ustc.edu.cn.

