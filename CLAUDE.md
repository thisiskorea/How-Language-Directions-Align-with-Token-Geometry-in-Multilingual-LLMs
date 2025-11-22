# CLAUDE.md - AI Assistant Guide

## Project Overview

This repository contains research code for analyzing how language directions align with token geometry in multilingual Large Language Models (LLMs). The project investigates language representation and classification across different layers of various LLMs using probing techniques.

**Research Focus:**
- Language classification using linear and MLP probes
- Token-language alignment analysis
- Probe geometry and manifold dimension estimation
- Generation-level validation across languages
- Multi-seed experiments for statistical robustness

**Supported Languages:** English, Spanish, Chinese, French, German

## Repository Structure

```
.
├── analyze_token_language_alignment (5).py  # Main analysis script
├── gpt_oss_20b_experiment.py               # GPT-OSS-20B specific experiments
├── multilingual_language_classification_multiseed.py  # Multi-model multi-seed experiments
├── probe_accuracies (2).csv                # Probe accuracy results
├── token_language_alignment (7).csv        # Token alignment analysis results
└── .git/                                   # Git repository
```

## Core Scripts

### 1. `analyze_token_language_alignment (5).py`

**Purpose:** Comprehensive token-language alignment analysis across all layers of LLMs.

**Key Features:**
- Extracts hidden states from all transformer layers
- Trains linear probes for 5-class language classification
- Analyzes how LM head tokens align with learned language directions
- Computes probe geometry (cosine similarity between language directions)
- Estimates manifold dimensions using PCA
- Performs generation-level validation
- Creates visualizations (heatmaps, line plots, scree plots)

**Configuration:**
- Models: Meta-Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-Math-7B, OpenMath2-Llama3.1-8B, OpenR1-Qwen-7B, GPT-OSS-20B
- Data: `/home/dilab05/work_directory/김재성/지티어트랙션_시험/ICLR/Data/multilingual_xnli_5lang.pkl`
- Output: `./token_language_alignment/`
- Batch sizes: Extract=8, Train=128
- Training: 3 epochs, lr=1e-3, weight_decay=0.01

**Key Functions:**
- `detect_token_language()`: Detects script/language of tokens using Unicode ranges
- `analyze_probe_geometry()`: Computes Gram matrix and cosine similarity between language directions
- `generation_level_validation()`: Analyzes next-token prediction entropy per language
- `estimate_manifold_dimensions()`: Computes intrinsic dimensionality using PCA
- `analyze_token_language_alignment()`: Main analysis computing token-to-language assignments

**Outputs:**
- `probe_accuracies.csv`: Model, layer, validation accuracy
- `token_language_alignment.csv`: Token counts, percentages, match rates per language
- `probe_geometry_stats.csv`: Orthogonality metrics
- `probe_geometry_metrics.csv`: Per-language weight magnitudes
- `generation_entropy_stats.csv`: Next-token entropy per language
- `manifold_dimension_estimates.csv`: Intrinsic dimensionality metrics
- Checkpoints: Saved in `checkpoints/{model_name}/layer_{idx}_probe.pt`

### 2. `gpt_oss_20b_experiment.py`

**Purpose:** Focused experiments on GPT-OSS-20B model with multi-seed robustness.

**Key Differences from main script:**
- Single model focus (openai/gpt-oss-20b)
- Multi-seed experiments (seeds: 42, 123, 456, 789, 2024)
- Both Linear and MLP probes
- Computes MLP-Linear gap to measure linearity
- Incremental saving with resumption support
- Per-language accuracy breakdown

**Outputs:**
- `all_seeds_results.csv`: Raw results for all seeds
- `aggregated_results_mean_std.csv`: Mean ± std across seeds

### 3. `multilingual_language_classification_multiseed.py`

**Purpose:** Multi-model, multi-seed experiments for comprehensive statistical analysis.

**Key Features:**
- Tests 5 different models with 5 random seeds each (25 total runs)
- Linear vs MLP probe comparison
- Per-language accuracy tracking
- Automatic resumption of incomplete experiments

**Models Tested:**
1. meta-llama/Llama-3.1-8B-Instruct
2. Qwen/Qwen2.5-7B-Instruct
3. Qwen/Qwen2.5-Math-7B-Instruct
4. nvidia/OpenMath2-Llama3.1-8B
5. open-r1/OpenR1-Qwen-7B

## Development Workflows

### Running Experiments

1. **Token-Language Alignment Analysis:**
```bash
conda activate only_for_VLLM
CUDA_VISIBLE_DEVICES=2 python "analyze_token_language_alignment (5).py"
```

2. **GPT-OSS-20B Specific:**
```bash
conda activate only_for_VLLM
CUDA_VISIBLE_DEVICES=3 python gpt_oss_20b_experiment.py
```

3. **Multi-Model Multi-Seed:**
```bash
conda activate only_for_VLLM
CUDA_VISIBLE_DEVICES=2,3 python multilingual_language_classification_multiseed.py
```

### Environment Setup

**Required Environment:** `only_for_VLLM`

**Key Dependencies:**
- PyTorch (with CUDA support)
- Transformers (HuggingFace)
- NumPy
- Pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

**GPU Requirements:**
- Minimum: 1 GPU with 24GB+ VRAM
- Recommended: 2 GPUs for parallel processing
- Models use float16 (Llama/Qwen) or bfloat16 (GPT-OSS)

### Data Format

**Input Data Structure** (`multilingual_xnli_5lang.pkl`):
```python
{
    'metadata': {
        'languages': [...],
        'num_classes': 5,
        'total_train': int,
        'total_val': int,
        'total_test': int
    },
    'train': {
        'texts': List[str],
        'labels': np.ndarray  # 0=EN, 1=ES, 2=ZH, 3=FR, 4=DE
    },
    'val': {...},
    'test': {...}
}
```

**Language Encoding:**
- 0: English
- 1: Spanish
- 2: Chinese
- 3: French
- 4: German

## Key Components

### Probe Models

**LinearProbe:**
```python
class LinearProbe(nn.Module):
    """LayerNorm + Linear classifier"""
    - norm: LayerNorm(hidden_size)
    - classifier: Linear(hidden_size, num_classes)
```

**MLPProbe:**
```python
class MLPProbe(nn.Module):
    """LayerNorm + 2-layer MLP with dropout"""
    - norm: LayerNorm(hidden_size)
    - fc1: Linear(hidden_size, 512)
    - dropout: Dropout(0.1)
    - fc2: Linear(512, num_classes)
```

### Feature Extraction

**Method:** Last token pooling from hidden states
- Applies chat template to inputs
- Extracts hidden states from all layers
- Uses final token representation
- Converts to float32 for compatibility

**Important:** All scripts apply chat templates to input texts before feature extraction for consistency with instruction-tuned models.

### Analysis Components

**1. Probe Geometry Analysis:**
- Computes Gram matrix (W @ W^T)
- Computes cosine similarity matrix (normalized)
- Measures orthogonality between language directions
- Tracks weight magnitudes per language

**2. Token-Language Alignment:**
- Computes cosine similarity: tokens × language_directions
- Assigns each token to nearest language direction
- Detects actual script of tokens using Unicode ranges
- Computes match rate: % of tokens matching expected language

**3. Manifold Dimension Estimation:**
- Applies PCA to language-specific token subsets
- Computes dimensions for 90%, 95%, 99% variance thresholds
- Calculates effective dimension using entropy
- Generates scree plots

**4. Generation Validation:**
- Computes next-token prediction entropy per language
- Analyzes top-k predictions for sample prompts
- Correlates entropy with probe accuracy

## Important Conventions

### File Naming
- Models use `/` in names → replaced with `_` for file paths
- Example: `meta-llama/Llama-3.1-8B-Instruct` → `meta-llama_Llama-3.1-8B-Instruct`

### Data Paths
- **Hardcoded paths:** Scripts reference `/home/dilab05/work_directory/...`
- **Action:** Update `DATA_PATH` constant when running on different machines

### GPU Configuration
- Set via `CUDA_VISIBLE_DEVICES` environment variable
- Main script: GPU 2
- GPT-OSS script: GPU 3
- Multi-seed script: GPUs 2,3

### Random Seeds
- **Reproducibility:** Seeds set for Python, NumPy, PyTorch, CUDA
- **Standard seeds:** [42, 123, 456, 789, 2024]
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

### Memory Management
- Features extracted once per model, then reused
- Models deleted after feature extraction
- `torch.cuda.empty_cache()` called regularly
- `gc.collect()` used for cleanup

## Output Files

### CSV Structure

**probe_accuracies.csv:**
```
model,layer,val_accuracy
meta-llama/Llama-3.1-8B-Instruct,0,20.0
meta-llama/Llama-3.1-8B-Instruct,1,99.82
```

**token_language_alignment.csv:**
```
model,layer,language,token_count,vocab_percentage,avg_similarity,match_rate_pct,match_count,total_sampled
meta-llama/Llama-3.1-8B-Instruct,0,english,46290,36.09,0.017532,71.12,32921,46290
```

**Checkpoint Structure:**
```python
checkpoint = {
    'probe_state_dict': {...},
    'classifier_weight': Tensor(num_classes, hidden_size),
    'classifier_bias': Tensor(num_classes),
    'layernorm_weight': Tensor(hidden_size),
    'layernorm_bias': Tensor(hidden_size),
    'hidden_size': int,
    'num_classes': int,
    'layer_idx': int,
    'model_name': str,
    'val_accuracy': float,
    'train_size': int,
    'val_size': int
}
```

## Common Patterns

### Loading Models
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or bfloat16 for GPT-OSS
    device_map="auto",
    trust_remote_code=True
)
```

### Chat Template Application
```python
messages = [{"role": "user", "content": text}]
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

### Feature Extraction Pattern
```python
outputs = model(**encoded, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
last_token_hidden = hidden_states[layer_idx][:, -1, :]  # Last token
```

### Probe Training Pattern
```python
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
# Train for 3 epochs, save best val accuracy checkpoint
```

## Troubleshooting

### Common Issues

1. **NaN in similarity matrix:**
   - Caused by zero-norm vectors
   - Scripts add epsilon (1e-8) during normalization
   - Affected layers are skipped

2. **BFloat16 conversion errors:**
   - Convert to float32 before numpy: `.float().cpu().numpy()`
   - GPT-OSS models require bfloat16 dtype

3. **Chat template failures:**
   - Some models may not have chat templates
   - Scripts include fallback to raw text

4. **Memory issues:**
   - Reduce `EXTRACT_BS` batch size
   - Process fewer layers at once
   - Delete model immediately after feature extraction

5. **Missing config attributes:**
   - Scripts check both `model.config` and `model.config.text_config`
   - Falls back to alternative attribute names (`num_hidden_layers` vs `num_layers`)

## Best Practices for AI Assistants

### When Modifying Code

1. **Preserve hardcoded paths:** Ask user for correct paths before changing
2. **Maintain seed consistency:** Use existing seed list for reproducibility
3. **Keep probe architecture:** Linear probe is standard; MLP for comparison
4. **Preserve output format:** CSV columns are expected by analysis scripts
5. **Test with single layer first:** Validate changes on layer 0 before full run

### When Adding Features

1. **Follow existing patterns:** Use similar structure to existing analysis functions
2. **Add to aggregation:** Update aggregation logic for new metrics
3. **Update visualizations:** Add corresponding plots if introducing new metrics
4. **Document in docstrings:** Follow numpy-style docstring format
5. **Save incremental results:** Enable resumption for long experiments

### When Debugging

1. **Check layer 0 first:** Often shows different behavior (random initialization)
2. **Validate shapes:** Print tensor shapes at each major step
3. **Monitor memory:** Use `nvidia-smi` during execution
4. **Check CSV outputs:** Verify data types and missing values
5. **Compare with existing results:** Use provided CSV files as reference

### Code Quality

1. **Language comments:** Mix of English docstrings and Korean comments is intentional
2. **Magic numbers:** Constants like 512 (MLP hidden), 0.1 (dropout) are standard
3. **Print statements:** Extensive logging is intentional for long-running experiments
4. **Error handling:** Basic try-except for model loading; propagate errors otherwise
5. **Type conversions:** Explicit float32 conversions are necessary, not redundant

## Experimental Design Notes

### Why Linear vs MLP Probes?

- **Linear probe:** Tests if language info is linearly accessible
- **MLP probe:** Measures if non-linear transformation helps
- **Gap metric:** MLP - Linear accuracy indicates non-linearity of representations

### Why Multi-Seed?

- Probe initialization is random
- Dataset shuffle order varies
- Statistical robustness through mean ± std reporting

### Why Last Token Pooling?

- Consistent with causal LM architecture
- Instruction-tuned models expect responses after prompts
- Chat template places user input before response

### Why All Layers?

- Tracks language information throughout network
- Early layers: Surface features
- Middle layers: Peak accuracy
- Late layers: Task-specific representations

## Research Context

This codebase supports research on:
- How multilingual models organize language information
- Geometric properties of language representations
- Relationship between token embeddings and language directions
- Linearity of language representations across layers
- Cross-model comparison of language encoding strategies

**Key Findings Expected:**
- High probe accuracy in middle layers
- Language-specific token clustering
- Varying degrees of orthogonality between language directions
- Different intrinsic dimensionalities per language
- Token-language alignment varies by model architecture

## Git Workflow

**Current Branch:** `claude/claude-md-miaffrkuro9sq7j1-01MyUJsRUaWEDSxmyPz7QZRn`

**Commit Strategy:**
- Descriptive messages for experimental results
- Include model name and layer range in commits
- Tag major experimental milestones

**Important:** Always push to the claude/* branch specified in the task context.

## Performance Optimization

### Speed Improvements
1. Extract features once, reuse for all probes
2. Use float16/bfloat16 for model inference
3. Increase batch size if memory allows
4. Process multiple GPUs in parallel
5. Cache tokenizer outputs when possible

### Memory Optimization
1. Delete models immediately after feature extraction
2. Process layers sequentially, not all at once
3. Use `torch.cuda.empty_cache()` liberally
4. Convert to numpy and delete tensors when done
5. Limit PCA components to 1000 for efficiency

## Contact and Support

For questions about the codebase or research methodology, refer to:
- Code comments (mix of English/Korean)
- Docstrings for function-level documentation
- Output CSVs for expected data formats
- This CLAUDE.md for high-level architecture

---

**Last Updated:** 2025-11-22
**Repository:** How-Language-Directions-Align-with-Token-Geometry-in-Multilingual-LLMs
