# How Language Directions Align with Token Geometry in Multilingual LLMs

[![WWW 2026](https://img.shields.io/badge/WWW-2026-blue)](https://www2026.thewebconf.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for the paper:

**"How Language Directions Align with Token Geometry in Multilingual LLMs"**
*JaeSeong Kim and Suan Lee*
The Web Conference (WWW) 2026, Dubai, UAE

---

## 📄 Abstract

Multilingual LLMs demonstrate strong performance across diverse languages, yet there has been limited systematic analysis of how language information is structured within their internal representation space and how it emerges across layers. We conduct a comprehensive probing study on six multilingual LLMs, covering **all 268 transformer layers**, using linear and nonlinear probes together with a new **Token-Language Alignment** analysis to quantify the layer-wise dynamics and geometric structure of language encoding.

### Key Findings

1. **Universal Linear Separability**: Language information becomes sharply separated in the first transformer block (+76.4±8.2%p from Layer 0→1) and remains almost fully linearly separable throughout model depth (99.8±0.1% accuracy, Linear-MLP gap: 0.58±0.12%p).

2. **Structural Imprinting**: The alignment between language directions and vocabulary embeddings is strongly tied to the language composition of the training data. Chinese-inclusive models achieve ZH Match@Peak of 16.43%, whereas English-centric models achieve only 3.90%, revealing a **4.21× structural imprinting effect**.

3. **Typological Dependencies**: Chinese reaches optimal separability in deeper layers (Layer 5.2±0.8), while Spanish and German converge earlier (Layer 2.5±0.4), indicating that non-alphabetic writing systems require deeper processing.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thisiskorea/How-Language-Directions-Align-with-Token-Geometry-in-Multilingual-LLMs.git
cd How-Language-Directions-Align-with-Token-Geometry-in-Multilingual-LLMs

# Create conda environment
conda create -n multilingual_analysis python=3.8
conda activate multilingual_analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- Transformers 4.30+
- CUDA-capable GPU with 24GB+ VRAM (recommended: 2x GPUs)

See `requirements.txt` for full dependency list.

---

## 📊 Experiments

Our analysis covers:
- **6 multilingual LLMs**: Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-Math-7B, OpenMath2-8B, OpenR1-7B, GPT-OSS-20B
- **268 transformer layers** across all models
- **5 languages**: English (EN), Spanish (ES), Chinese (ZH), French (FR), German (DE)
- **2,680 independent experiments** with multiple random seeds

### Models Tested

| Model | Parameters | Language Focus | Layers |
|-------|-----------|----------------|--------|
| Llama-3.1-8B-Instruct | 8B | English-centric | 32 |
| Qwen2.5-7B-Instruct | 7B | Chinese-inclusive | 28 |
| Qwen2.5-Math-7B-Instruct | 7B | Chinese-inclusive | 28 |
| OpenMath2-Llama3.1-8B | 8B | English-centric | 32 |
| OpenR1-Qwen-7B | 7B | Chinese-inclusive | 28 |
| GPT-OSS-20B | 20B | Balanced | 48 |

---

## 🔧 Usage

### 1. Data Preparation

Download the XNLI multilingual dataset:

```bash
# Download XNLI dataset
# Place the processed data at: ./data/multilingual_xnli_5lang.pkl
# Or update DATA_PATH in the scripts
```

**Data Format**: The dataset should be a pickle file containing:
```python
{
    'metadata': {
        'languages': ['en', 'es', 'zh', 'fr', 'de'],
        'num_classes': 5,
        'total_train': 25000,  # 5000 per language
        'total_val': 12500     # 2500 per language
    },
    'train': {'texts': [...], 'labels': [...]},
    'val': {'texts': [...], 'labels': [...]}
}
```

### 2. Running Experiments

#### Main Analysis (All Models, All Layers)

```bash
conda activate multilingual_analysis
CUDA_VISIBLE_DEVICES=0 python "analyze_token_language_alignment (5).py"
```

This script:
- Extracts hidden states from all transformer layers
- Trains linear probes for language classification
- Analyzes token-language alignment
- Computes probe geometry and manifold dimensions
- Generates visualizations

**Outputs**:
- `./token_language_alignment/probe_accuracies.csv`
- `./token_language_alignment/token_language_alignment.csv`
- `./token_language_alignment/probe_geometry_stats.csv`
- `./token_language_alignment/manifold_dimension_estimates.csv`
- `./token_language_alignment/checkpoints/{model}/layer_{idx}_probe.pt`

#### GPT-OSS-20B Focused Analysis

```bash
CUDA_VISIBLE_DEVICES=0 python gpt_oss_20b_experiment.py
```

Multi-seed experiments (seeds: 42, 123, 456, 789, 2024) for statistical robustness.

**Outputs**:
- `./gpt_oss_20b_results/all_seeds_results.csv`
- `./gpt_oss_20b_results/aggregated_results_mean_std.csv`

#### Multi-Model Multi-Seed Experiments

```bash
CUDA_VISIBLE_DEVICES=0,1 python multilingual_language_classification_multiseed.py
```

Tests 5 models with 5 random seeds each (25 total runs).

**Outputs**:
- `./multilingual_experiments_xnli_multiseed/all_seeds_results.csv`
- `./multilingual_experiments_xnli_multiseed/aggregated_results_mean_std.csv`

---

## 📁 Repository Structure

```
.
├── analyze_token_language_alignment (5).py   # Main comprehensive analysis
├── gpt_oss_20b_experiment.py                # GPT-OSS-20B focused experiments
├── multilingual_language_classification_multiseed.py  # Multi-model experiments
├── probe_accuracies (2).csv                 # Example probe accuracy results
├── token_language_alignment (7).csv         # Example alignment results
├── requirements.txt                         # Python dependencies
├── README.md                               # This file
├── CLAUDE.md                               # Technical documentation for AI assistants
└── LICENSE                                 # MIT License
```

---

## 📈 Results

### Probe Accuracy by Layer

All models achieve near-ceiling accuracy (>99%) after Layer 1, with minimal Linear-MLP gap:

| Model | Avg Linear Acc | Avg MLP Acc | Gap | Layer 0→1 Jump |
|-------|---------------|-------------|-----|----------------|
| Llama-3.1-8B | 96.2±1.8% | 96.0±1.8% | -0.2%p | +79.8%p |
| Qwen2.5-7B | 95.2±1.5% | 94.8±1.7% | -0.5%p | +80.7%p |
| OpenR1-7B | 94.0±1.6% | 93.9±2.5% | -0.1%p | +74.3%p |

### Token-Language Alignment (Match@Peak)

English-centric vs Chinese-inclusive models show stark differences:

| Language | English-centric Models | Chinese-inclusive Models | Ratio |
|----------|----------------------|------------------------|-------|
| EN | 69.05% | 54.13% | 1.28× |
| **ZH** | **3.90%** | **16.43%** | **4.21×** |
| ES | 1.60% | 0.90% | 1.78× |
| FR | 0.85% | 0.80% | 1.06× |
| DE | 0.40% | 0.30% | 1.33× |

This demonstrates **structural imprinting**: pretraining data distribution shapes the geometry of internal representations.

---

## 🔬 Methodology

### Linear and Nonlinear Probing

For each layer ℓ, we extract the final token's hidden state and train:

1. **Linear Probe**: `f_lin(h) = W_c · LN(h) + b_c`
2. **MLP Probe**: `f_mlp(h) = W_2 · ReLU(W_1 · LN(h))`

Both use LayerNorm to remove inter-layer scale differences and measure pure linear separability.

### Token-Language Alignment

We compute cosine similarity between:
- Probe-learned language directions `w_L^(ℓ)`
- LM head vocabulary embeddings `e_v`

**Metrics**:
- **PeakDepth**: Normalized layer where language L is most expressed
- **PeakVocab**: Maximum vocabulary share for language L
- **Match@Peak**: % of assigned tokens whose decoded text matches language L

This quantifies how pretraining data structure is "imprinted" into representation geometry.

---

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{kim2026language,
  title={How Language Directions Align with Token Geometry in Multilingual LLMs},
  author={Kim, JaeSeong and Lee, Suan},
  booktitle={Proceedings of the ACM Web Conference 2026},
  year={2026},
  location={Dubai, United Arab Emirates},
  publisher={ACM}
}
```

---

## 🛠️ Configuration

### GPU Settings

- **Main analysis**: 1 GPU (24GB+ VRAM)
- **Multi-seed experiments**: 2 GPUs recommended
- Set via `CUDA_VISIBLE_DEVICES`

### Data Paths

Update `DATA_PATH` in each script to point to your dataset location:

```python
DATA_PATH = "/path/to/your/multilingual_xnli_5lang.pkl"
```

### Hyperparameters

Default settings (optimized for reproducibility):
- **Feature extraction batch size**: 8
- **Probe training batch size**: 128
- **Learning rate**: 1e-3
- **Weight decay**: 0.01
- **Epochs**: 3 (with early stopping)
- **Seeds**: [42, 123, 456, 789, 2024]

---

## 🐛 Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `EXTRACT_BS` from 8 to 4 or 2
2. **Missing chat template**: Some models may not have chat templates; script falls back to raw text
3. **BFloat16 errors**: GPT-OSS models require `torch_dtype=torch.bfloat16`
4. **NaN in similarity matrix**: Script automatically skips affected layers

### System Requirements

- Minimum: 1x NVIDIA GPU with 24GB VRAM
- Recommended: 2x A100 40GB or V100 32GB
- Disk space: ~50GB for model weights + checkpoints

---

## 📧 Contact

- **JaeSeong Kim**: mmmqp1010@gmail.com
- **Suan Lee**: suanlee@semyung.ac.kr

---

## 🙏 Acknowledgments

This research was conducted at Semyung University. We thank the open-source community for providing the pretrained models and datasets used in this study.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Paper**: [WWW 2026 Proceedings](https://www2026.thewebconf.org/) (to be published)
- **Models**: [Hugging Face Hub](https://huggingface.co/)
- **Dataset**: [XNLI](https://github.com/facebookresearch/XNLI)

---

**Note**: This repository contains the official implementation for reproducing all experiments and figures in our WWW 2026 paper. For questions about the methodology or results, please refer to the paper or contact the authors.
