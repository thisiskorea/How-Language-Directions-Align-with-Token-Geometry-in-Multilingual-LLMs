"""
Token-Language Alignment Analysis

Analyzes how LM head tokens align with learned language directions from probing experiments.

For each layer's language probe:
1. Extract probe weight vectors (5 language directions)
2. Extract LM head token vectors (vocab_size tokens)
3. Compute cosine similarity between each token and each language direction
4. Identify top-N tokens most aligned with each language
5. Compute language distribution across vocabulary

Usage:
    conda activate only_for_VLLM
    CUDA_VISIBLE_DEVICES=2 python analyze_token_language_alignment.py

Environment:
    GPU: 2
    Environment: only_for_VLLM
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_LIST = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "nvidia/OpenMath2-Llama3.1-8B",
    "open-r1/OpenR1-Qwen-7B",
    "openai/gpt-oss-20b"
]

DATA_PATH = "/home/dilab05/work_directory/김재성/지티어트랙션_시험/ICLR/Data/multilingual_xnli_5lang.pkl"
OUTPUT_DIR = "./token_language_alignment"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Hyperparameters
EXTRACT_BS = 8
TRAIN_BS = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 3
MAX_LENGTH = 256
TOP_N_TOKENS = None  # None = analyze ALL tokens assigned to each language (slower but complete)

# GPU Configuration
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Language mapping
LANGUAGE_NAMES = ['english', 'spanish', 'chinese', 'french', 'german']
LANGUAGE_CODES = ['en', 'es', 'zh', 'fr', 'de']


# ============================================================================
# Language Detection Functions
# ============================================================================

def detect_token_language(token_text):
    """
    Detect language/script of a token based on character patterns

    Returns one of:
    - 'english', 'spanish', 'french', 'german', 'chinese'
    - 'korean', 'cyrillic', 'arabic', 'thai', 'greek', 'hebrew'
    - 'code', 'number', 'symbol', 'mixed', 'unknown'
    """
    if not token_text or len(token_text.strip()) == 0:
        return 'empty'

    text = token_text.strip()

    # Check for code patterns (common programming tokens)
    code_patterns = ['.', '_', '(', ')', '{', '}', '[', ']', '/', '\\', '<', '>', '=', ';', ':', ',']
    if any(text.startswith(p) or text.endswith(p) for p in code_patterns):
        if sum(c.isalpha() for c in text) < len(text) * 0.3:  # Less than 30% alphabetic
            return 'code'

    # Count characters by script
    char_counts = {
        'cjk': 0,           # Chinese, Japanese, Korean ideographs
        'korean': 0,        # Hangul
        'cyrillic': 0,      # Russian, etc.
        'arabic': 0,        # Arabic script
        'thai': 0,          # Thai script
        'greek': 0,         # Greek
        'hebrew': 0,        # Hebrew
        'latin': 0,         # Latin-based (English, Spanish, French, German)
        'number': 0,        # Digits
        'symbol': 0         # Other symbols
    }

    # Spanish-specific characters
    spanish_chars = set('áéíóúñÁÉÍÓÚÑ¿¡')
    # French-specific characters
    french_chars = set('àâçèêëîïôùûüÿœæÀÂÇÈÊËÎÏÔÙÛÜŸŒÆ')
    # German-specific characters
    german_chars = set('äöüßÄÖÜ')

    has_spanish = False
    has_french = False
    has_german = False

    for char in text:
        code_point = ord(char)

        # CJK Ideographs (Chinese characters)
        if (0x4E00 <= code_point <= 0x9FFF or      # CJK Unified Ideographs
            0x3400 <= code_point <= 0x4DBF or      # CJK Extension A
            0x20000 <= code_point <= 0x2A6DF):     # CJK Extension B
            char_counts['cjk'] += 1
        # Hangul (Korean)
        elif (0xAC00 <= code_point <= 0xD7AF or    # Hangul Syllables
              0x1100 <= code_point <= 0x11FF):      # Hangul Jamo
            char_counts['korean'] += 1
        # Cyrillic
        elif 0x0400 <= code_point <= 0x04FF:
            char_counts['cyrillic'] += 1
        # Arabic
        elif 0x0600 <= code_point <= 0x06FF:
            char_counts['arabic'] += 1
        # Thai
        elif 0x0E00 <= code_point <= 0x0E7F:
            char_counts['thai'] += 1
        # Greek
        elif 0x0370 <= code_point <= 0x03FF:
            char_counts['greek'] += 1
        # Hebrew
        elif 0x0590 <= code_point <= 0x05FF:
            char_counts['hebrew'] += 1
        # Latin (basic + extended)
        elif ((0x0041 <= code_point <= 0x005A) or  # A-Z
              (0x0061 <= code_point <= 0x007A) or  # a-z
              (0x00C0 <= code_point <= 0x00FF) or  # Latin-1 Supplement
              (0x0100 <= code_point <= 0x017F)):   # Latin Extended-A
            char_counts['latin'] += 1
            # Check for language-specific diacritics
            if char in spanish_chars:
                has_spanish = True
            if char in french_chars:
                has_french = True
            if char in german_chars:
                has_german = True
        # Numbers
        elif char.isdigit():
            char_counts['number'] += 1
        # Symbols
        else:
            char_counts['symbol'] += 1

    total_chars = sum(char_counts.values())
    if total_chars == 0:
        return 'empty'

    # Determine dominant script
    max_script = max(char_counts.items(), key=lambda x: x[1])
    dominant_script = max_script[0]
    dominant_ratio = max_script[1] / total_chars

    # If numbers/symbols dominate
    if char_counts['number'] > total_chars * 0.7:
        return 'number'
    if char_counts['symbol'] > total_chars * 0.5:
        return 'symbol'

    # Script-based classification
    if dominant_ratio < 0.5:
        return 'mixed'

    if dominant_script == 'cjk':
        return 'chinese'
    elif dominant_script == 'korean':
        return 'korean'
    elif dominant_script == 'cyrillic':
        return 'cyrillic'
    elif dominant_script == 'arabic':
        return 'arabic'
    elif dominant_script == 'thai':
        return 'thai'
    elif dominant_script == 'greek':
        return 'greek'
    elif dominant_script == 'hebrew':
        return 'hebrew'
    elif dominant_script == 'latin':
        # Distinguish between Latin-based languages
        if has_spanish:
            return 'spanish'
        elif has_french:
            return 'french'
        elif has_german:
            return 'german'
        else:
            # Default to English for plain Latin without special chars
            return 'english'

    return 'unknown'


def calculate_language_match_rate(top_tokens_df, expected_language):
    """
    Calculate how many of the top tokens actually match the expected language

    Args:
        top_tokens_df: DataFrame with 'token_text' and 'detected_language' columns
        expected_language: str, one of LANGUAGE_NAMES

    Returns:
        dict with statistics
    """
    detected_languages = top_tokens_df['detected_language'].value_counts()
    total = len(top_tokens_df)

    match_count = (top_tokens_df['detected_language'] == expected_language).sum()
    match_rate = match_count / total if total > 0 else 0.0

    return {
        'total_tokens': total,
        'match_count': match_count,
        'match_rate': match_rate,
        'detected_distribution': detected_languages.to_dict()
    }


# ============================================================================
# Probe Geometry Analysis
# ============================================================================

def analyze_probe_geometry(probe, layer_idx, model_name):
    """
    Analyze geometric properties of language probe directions

    Computes:
    1. Gram matrix (5×5): pairwise dot products
    2. Cosine similarity matrix (5×5): normalized pairwise similarities
    3. Orthogonality metrics: how perpendicular are language directions?
    4. Clustering: do Romance languages cluster together vs Chinese?

    Returns:
        dict with keys:
        - 'gram_matrix_df': DataFrame with Gram matrix
        - 'cosine_matrix_df': DataFrame with cosine similarity matrix
        - 'orthogonality_stats': Summary statistics
        - 'geometry_metrics': Per-language metrics
    """
    print(f"\n{'='*70}")
    print(f"PROBE GEOMETRY ANALYSIS - Layer {layer_idx}")
    print(f"{'='*70}")

    # Extract probe weights: (5, hidden_size)
    probe_weight = probe.classifier.weight.data.cpu().float()

    # 1. Gram Matrix: W @ W^T (unnormalized)
    gram_matrix = (probe_weight @ probe_weight.T).numpy()

    # 2. Cosine Similarity Matrix (normalized)
    probe_weight_norm = F.normalize(probe_weight, dim=1)
    cosine_matrix = (probe_weight_norm @ probe_weight_norm.T).numpy()

    # 3. Convert to DataFrames for readability
    gram_df = pd.DataFrame(
        gram_matrix,
        index=LANGUAGE_NAMES,
        columns=LANGUAGE_NAMES
    )

    cosine_df = pd.DataFrame(
        cosine_matrix,
        index=LANGUAGE_NAMES,
        columns=LANGUAGE_NAMES
    )

    # 4. Compute orthogonality metrics
    # Extract off-diagonal elements (exclude self-similarity)
    off_diag_mask = ~np.eye(5, dtype=bool)
    off_diag_cosines = cosine_matrix[off_diag_mask]

    orthogonality_stats = {
        'model': model_name,
        'layer': layer_idx,
        'mean_off_diag_cosine': float(off_diag_cosines.mean()),
        'std_off_diag_cosine': float(off_diag_cosines.std()),
        'max_off_diag_cosine': float(off_diag_cosines.max()),
        'min_off_diag_cosine': float(off_diag_cosines.min()),
        'mean_abs_cosine': float(np.abs(off_diag_cosines).mean()),
    }

    # 5. Language-specific metrics
    geometry_metrics = []
    for i, lang in enumerate(LANGUAGE_NAMES):
        # Magnitude (L2 norm)
        magnitude = float(np.linalg.norm(probe_weight[i].numpy()))

        # Average similarity to other languages
        other_similarities = np.delete(cosine_matrix[i], i)
        avg_sim_to_others = float(other_similarities.mean())
        max_sim_to_others = float(other_similarities.max())

        geometry_metrics.append({
            'model': model_name,
            'layer': layer_idx,
            'language': lang,
            'weight_magnitude': magnitude,
            'avg_cosine_to_others': avg_sim_to_others,
            'max_cosine_to_others': max_sim_to_others
        })

    # 6. Print summary
    print("\n[Cosine Similarity Matrix]")
    print(cosine_df.round(4))

    print("\n[Orthogonality Statistics]")
    print(f"  Mean off-diagonal cosine: {orthogonality_stats['mean_off_diag_cosine']:.4f}")
    print(f"  Std off-diagonal cosine: {orthogonality_stats['std_off_diag_cosine']:.4f}")
    print(f"  Range: [{orthogonality_stats['min_off_diag_cosine']:.4f}, {orthogonality_stats['max_off_diag_cosine']:.4f}]")

    print("\n[Language-Specific Metrics]")
    for metric in geometry_metrics:
        print(f"  {metric['language']:10s}: magnitude={metric['weight_magnitude']:.2f}, avg_sim={metric['avg_cosine_to_others']:.4f}")

    return {
        'gram_matrix_df': gram_df,
        'cosine_matrix_df': cosine_df,
        'orthogonality_stats': orthogonality_stats,
        'geometry_metrics': geometry_metrics
    }


# ============================================================================
# Generation-level Validation
# ============================================================================

def generation_level_validation(model, tokenizer, probe, layer_idx, model_name, num_samples=10):
    """
    Validate probe findings through generation-level analysis

    For each language:
    1. Prepare sample prompts in that language
    2. Measure next-token prediction entropy: H(next_token | context)
    3. Compute perplexity for continuation
    4. Analyze if high entropy correlates with low probe accuracy

    Returns:
        dict with keys:
        - 'entropy_stats': DataFrame with per-language entropy
        - 'top_predictions': DataFrame with top-k predictions per language
    """
    print(f"\n{'='*70}")
    print(f"GENERATION-LEVEL VALIDATION - Layer {layer_idx}")
    print(f"{'='*70}")

    # Sample prompts for each language
    sample_prompts = {
        'english': [
            "Hello, how are", "The quick brown", "In the morning",
            "Today's weather is", "I would like to", "What is your",
            "Please tell me", "Thank you for", "Can you help", "This is very"
        ],
        'spanish': [
            "Hola, ¿cómo estás", "Buenos días, me", "Por favor, dime",
            "Muchas gracias por", "¿Qué tal tu", "Me gustaría saber",
            "La verdad es que", "En mi opinión", "¿Puedes ayudarme", "Esto es muy"
        ],
        'chinese': [
            "你好，最近怎么", "今天天气很", "我想知道",
            "请告诉我", "非常感谢", "这是什么",
            "我们应该", "你可以", "明天我要", "这个问题"
        ],
        'french': [
            "Bonjour, comment allez", "S'il vous plaît", "Merci beaucoup pour",
            "Je voudrais savoir", "Qu'est-ce que", "C'est très",
            "Aujourd'hui nous", "Pouvez-vous m'aider", "Il est important", "Dans ce cas"
        ],
        'german': [
            "Guten Tag, wie", "Vielen Dank für", "Bitte sagen Sie",
            "Was ist das", "Ich möchte wissen", "Können Sie mir",
            "Das ist sehr", "Heute wollen wir", "In diesem Fall", "Meiner Meinung nach"
        ]
    }

    model.eval()
    entropy_results = []
    top_predictions_results = []

    for lang_name in LANGUAGE_NAMES:
        prompts = sample_prompts[lang_name][:num_samples]
        lang_entropies = []

        print(f"\n[Analyzing {lang_name}...]")

        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

            with torch.no_grad():
                # Get next-token logits
                outputs = model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)

                # Compute probability distribution
                probs = F.softmax(next_token_logits, dim=-1)

                # Compute entropy: H = -Σ p(x) log p(x)
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)

                lang_entropies.append(entropy.item())

                # Get top-5 predicted tokens
                top_k = 5
                top_probs, top_indices = torch.topk(probs[0], k=top_k)

                for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
                    token_text = tokenizer.decode([idx.item()])
                    top_predictions_results.append({
                        'model': model_name,
                        'layer': layer_idx,
                        'language': lang_name,
                        'prompt': prompt,
                        'rank': rank,
                        'token_id': int(idx.item()),
                        'token_text': token_text,
                        'probability': float(prob.item())
                    })

        # Compute statistics
        mean_entropy = np.mean(lang_entropies)
        std_entropy = np.std(lang_entropies)

        entropy_results.append({
            'model': model_name,
            'layer': layer_idx,
            'language': lang_name,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'min_entropy': min(lang_entropies),
            'max_entropy': max(lang_entropies),
            'num_samples': len(prompts)
        })

        print(f"  Mean entropy: {mean_entropy:.4f} ± {std_entropy:.4f}")

    entropy_df = pd.DataFrame(entropy_results)
    top_predictions_df = pd.DataFrame(top_predictions_results)

    # Print summary comparison
    print(f"\n[Entropy Comparison - Layer {layer_idx}]")
    print(entropy_df[['language', 'mean_entropy', 'std_entropy']].to_string(index=False))

    return {
        'entropy_stats': entropy_df,
        'top_predictions': top_predictions_df
    }


# ============================================================================
# Manifold Dimension Estimation
# ============================================================================

def estimate_manifold_dimensions(lm_weight, token_assignments, layer_idx, model_name):
    """
    Estimate intrinsic dimensionality of language-specific token manifolds

    For each language:
    1. Extract embeddings of all tokens assigned to that language
    2. Apply PCA and compute cumulative variance explained
    3. Estimate intrinsic dimension (# components explaining 90%, 95%, 99% variance)
    4. Compare EN vs ZH vs other languages

    Returns:
        dict with keys:
        - 'dimension_estimates': DataFrame with dimension metrics
        - 'variance_explained': DataFrame with PCA variance ratios
        - 'pca_objects': Dict of fitted PCA objects for further analysis
    """
    print(f"\n{'='*70}")
    print(f"MANIFOLD DIMENSION ESTIMATION - Layer {layer_idx}")
    print(f"{'='*70}")

    dimension_results = []
    variance_results = []
    pca_objects = {}

    # Convert to numpy if tensor
    if isinstance(lm_weight, torch.Tensor):
        lm_weight = lm_weight.cpu().float().numpy()

    for lang_idx, lang_name in enumerate(LANGUAGE_NAMES):
        # Get tokens assigned to this language
        assigned_indices = np.where(token_assignments == lang_idx)[0]
        num_tokens = len(assigned_indices)

        print(f"\n[Analyzing {lang_name}: {num_tokens} tokens]")

        if num_tokens < 10:
            print(f"  Warning: Too few tokens ({num_tokens}), skipping...")
            continue

        # Extract embeddings
        lang_embeddings = lm_weight[assigned_indices]  # (num_tokens, hidden_size)

        # Fit PCA (use min of num_tokens or hidden_size components)
        n_components = min(num_tokens - 1, lm_weight.shape[1], 1000)  # Cap at 1000 for efficiency

        pca = PCA(n_components=n_components)
        pca.fit(lang_embeddings)

        # Store PCA object
        pca_objects[lang_name] = pca

        # Compute cumulative variance explained
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find dimensions for different variance thresholds
        dim_90 = np.argmax(cumsum_variance >= 0.90) + 1
        dim_95 = np.argmax(cumsum_variance >= 0.95) + 1
        dim_99 = np.argmax(cumsum_variance >= 0.99) + 1

        # Effective dimension (Shannon entropy of variance ratios)
        variance_ratios = pca.explained_variance_ratio_
        variance_ratios = variance_ratios[variance_ratios > 1e-10]  # Filter near-zero
        normalized_ratios = variance_ratios / variance_ratios.sum()
        effective_dim = np.exp(-np.sum(normalized_ratios * np.log(normalized_ratios + 1e-10)))

        dimension_results.append({
            'model': model_name,
            'layer': layer_idx,
            'language': lang_name,
            'num_tokens': num_tokens,
            'embedding_dim': lm_weight.shape[1],
            'dim_90pct': int(dim_90),
            'dim_95pct': int(dim_95),
            'dim_99pct': int(dim_99),
            'effective_dim': float(effective_dim),
            'first_pc_variance': float(pca.explained_variance_ratio_[0]),
            'top10_pc_variance': float(cumsum_variance[min(9, len(cumsum_variance)-1)])
        })

        # Store variance explained for first 50 components
        for pc_idx in range(min(50, n_components)):
            variance_results.append({
                'model': model_name,
                'layer': layer_idx,
                'language': lang_name,
                'component': pc_idx + 1,
                'variance_ratio': float(pca.explained_variance_ratio_[pc_idx]),
                'cumulative_variance': float(cumsum_variance[pc_idx])
            })

        print(f"  Intrinsic dimensions:")
        print(f"    90% variance: {dim_90} dims")
        print(f"    95% variance: {dim_95} dims")
        print(f"    99% variance: {dim_99} dims")
        print(f"    Effective dim: {effective_dim:.1f}")
        print(f"  First PC explains: {pca.explained_variance_ratio_[0]*100:.2f}% variance")

    dimension_df = pd.DataFrame(dimension_results)
    variance_df = pd.DataFrame(variance_results)

    # Print comparison
    print(f"\n[Dimension Comparison - Layer {layer_idx}]")
    if len(dimension_df) > 0:
        print(dimension_df[['language', 'num_tokens', 'dim_90pct', 'dim_95pct', 'effective_dim']].to_string(index=False))

    return {
        'dimension_estimates': dimension_df,
        'variance_explained': variance_df,
        'pca_objects': pca_objects
    }


# ============================================================================
# Dataset
# ============================================================================

class LanguageDataset(Dataset):
    """Language classification dataset"""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }


class FeatureDataset(Dataset):
    """Feature vector dataset"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# Probe Model
# ============================================================================

class LinearProbe(nn.Module):
    """Linear probe: LayerNorm + Linear"""

    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.norm(x)
        return self.classifier(x)


# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load data"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    metadata = data['metadata']
    print(f"Languages: {metadata['languages']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Train samples: {metadata['total_train']}")
    print(f"Val samples: {metadata['total_val']}")

    # Test dataset
    if 'test' in data:
        print(f"Test samples: {metadata['total_test']}")
        test_texts = data['test']['texts']
        test_labels = data['test']['labels']
    else:
        print(f"Test samples: {metadata['total_val']} (using val as test)")
        test_texts = data['val']['texts']
        test_labels = data['val']['labels']

    train_dataset = LanguageDataset(
        data['train']['texts'],
        data['train']['labels']
    )
    val_dataset = LanguageDataset(
        data['val']['texts'],
        data['val']['labels']
    )
    test_dataset = LanguageDataset(
        test_texts,
        test_labels
    )

    return train_dataset, val_dataset, test_dataset, metadata


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(model, tokenizer, dataset, batch_size=EXTRACT_BS):
    """Extract hidden states from all layers"""
    print("\nExtracting features from all layers...")
    print("NOTE: Applying chat template to input texts")

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    config = getattr(model.config, 'text_config', model.config)
    num_layers = getattr(config, 'num_hidden_layers',
                        getattr(config, 'num_layers', None))
    if num_layers is None:
        raise ValueError(f"Could not find num_layers attribute in model config")

    all_features = {layer_idx: [] for layer_idx in range(num_layers)}
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Feature extraction"):
            texts = batch['text']
            labels = batch['label'].numpy()

            # Apply chat template
            formatted_texts = []
            for text in texts:
                messages = [{"role": "user", "content": text}]
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)

            # Tokenize
            encoded = tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(DEVICE)

            # Forward pass
            outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Last token pooling
            for layer_idx in range(num_layers):
                layer_hidden = hidden_states[layer_idx]
                last_token_hidden = layer_hidden[:, -1, :]
                # Convert to float32 before numpy (for bfloat16 compatibility)
                all_features[layer_idx].append(last_token_hidden.cpu().float().numpy())

            all_labels.append(labels)

    # Concatenate
    for layer_idx in range(num_layers):
        all_features[layer_idx] = np.vstack(all_features[layer_idx])

    all_labels = np.concatenate(all_labels)

    print(f"✓ Extracted features from {num_layers} layers")
    print(f"  Feature shape per layer: {all_features[0].shape}")
    print(f"  Labels shape: {all_labels.shape}")

    return all_features, all_labels


# ============================================================================
# Probe Training
# ============================================================================

def train_probe(probe, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train probe"""
    probe = probe.to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        probe.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        probe.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                logits = probe(features)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(labels.numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = probe.state_dict().copy()

    # Load best
    probe.load_state_dict(best_state)

    return probe, best_val_acc


# ============================================================================
# Token-Language Alignment Analysis
# ============================================================================

def analyze_token_language_alignment(model, tokenizer, probe, layer_idx, model_name):
    """
    Analyze alignment between LM head tokens and language probe directions

    Returns:
        dict with keys:
        - 'top_tokens': DataFrame with top-N tokens per language
        - 'distribution': DataFrame with language distribution
        - 'similarity_matrix': (vocab_size, num_languages) similarity matrix
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING TOKEN-LANGUAGE ALIGNMENT - Layer {layer_idx}")
    print(f"{'='*70}")

    # Extract probe weight (language direction vectors)
    probe_weight = probe.classifier.weight.data  # (num_classes=5, hidden_size)
    print(f"Probe weight shape: {probe_weight.shape}")

    # Extract LM head weight (token embedding vectors)
    lm_head = model.get_output_embeddings()
    lm_weight = lm_head.weight.data  # (vocab_size, hidden_size)
    print(f"LM head weight shape: {lm_weight.shape}")
    vocab_size = lm_weight.shape[0]

    # Convert to same dtype (float32) for cosine similarity computation
    # Move to CPU first to avoid dtype issues with different device dtypes
    probe_weight = probe_weight.cpu().float()
    lm_weight = lm_weight.cpu().float()

    # Normalize vectors for cosine similarity with small epsilon to avoid division by zero
    probe_weight_norm = F.normalize(probe_weight + 1e-8, dim=1)  # (5, hidden_size)
    lm_weight_norm = F.normalize(lm_weight + 1e-8, dim=1)        # (vocab_size, hidden_size)

    # Compute similarity matrix: (vocab_size, 5)
    # Each row: token's similarity to each of 5 language directions
    similarity_matrix = lm_weight_norm @ probe_weight_norm.T
    similarity_matrix = similarity_matrix.numpy()

    # Check for NaN values
    if np.isnan(similarity_matrix).any():
        print(f"  Warning: NaN detected in similarity matrix for layer {layer_idx}")
        print(f"  Skipping this layer...")
        return None

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

    # First, compute language distribution (which language each token is most aligned with)
    print(f"\n[Computing token-language assignments...]")
    token_language_assignments = np.argmax(similarity_matrix, axis=1)
    language_counts = np.bincount(token_language_assignments, minlength=5)

    print(f"Token assignments: {language_counts}")

    # Estimate manifold dimensions for each language
    manifold_results = estimate_manifold_dimensions(
        lm_weight, token_language_assignments, layer_idx, model_name
    )

    # Now analyze ALL tokens assigned to each language
    all_tokens_data = []

    for lang_idx, lang_name in enumerate(LANGUAGE_NAMES):
        # Get ALL token indices assigned to this language
        assigned_indices = np.where(token_language_assignments == lang_idx)[0]
        num_assigned = len(assigned_indices)

        print(f"  Processing {lang_name}: {num_assigned} tokens assigned...")

        # Get similarity scores for assigned tokens
        assigned_similarities = similarity_matrix[assigned_indices, lang_idx]

        # Sort by similarity (highest first) for ranking
        sorted_order = np.argsort(assigned_similarities)[::-1]
        sorted_indices = assigned_indices[sorted_order]
        sorted_scores = assigned_similarities[sorted_order]

        # Decode and detect language for ALL assigned tokens
        for rank, (token_id, score) in enumerate(zip(sorted_indices, sorted_scores)):
            try:
                token_text = tokenizer.decode([token_id])
            except:
                token_text = f"<DECODE_ERROR_{token_id}>"

            # Detect actual language of the token
            detected_lang = detect_token_language(token_text)

            all_tokens_data.append({
                'model': model_name,
                'layer': layer_idx,
                'language': lang_name,
                'rank': rank + 1,
                'token_id': int(token_id),
                'token_text': token_text,
                'similarity': float(score),
                'detected_language': detected_lang
            })

    top_tokens_df = pd.DataFrame(all_tokens_data)

    distribution_data = []
    for lang_idx, lang_name in enumerate(LANGUAGE_NAMES):
        count = int(language_counts[lang_idx])
        percentage = (count / vocab_size) * 100  # Already in percentage
        avg_similarity = float(similarity_matrix[token_language_assignments == lang_idx, lang_idx].mean())

        distribution_data.append({
            'model': model_name,
            'layer': layer_idx,
            'language': lang_name,
            'token_count': count,
            'vocab_percentage': round(percentage, 2),  # Rename to vocab_percentage for clarity
            'avg_similarity': round(avg_similarity, 6)
        })

    distribution_df = pd.DataFrame(distribution_data)

    # Print summary
    print(f"\n[Language Distribution - Layer {layer_idx}]")
    print(distribution_df[['language', 'token_count', 'vocab_percentage', 'avg_similarity']].to_string(index=False))

    # Calculate language match rates
    print(f"\n[Language Match Rates - Layer {layer_idx}]")
    print("(Percentage of ALL assigned tokens that match the expected language)")
    for lang_name in LANGUAGE_NAMES:
        lang_tokens = top_tokens_df[top_tokens_df['language'] == lang_name]
        match_stats = calculate_language_match_rate(lang_tokens, lang_name)
        match_pct = match_stats['match_rate'] * 100
        print(f"  {lang_name:10s}: {match_pct:5.1f}% ({match_stats['match_count']:3d}/{match_stats['total_tokens']:3d} tokens)")

        # Show top-3 detected languages for this direction
        det_dist = match_stats['detected_distribution']
        top_detected = sorted(det_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        det_str = ", ".join([f"{lang}:{count}" for lang, count in top_detected])
        print(f"              Distribution: {det_str}")

    print(f"\n[Top-5 Tokens per Language - Layer {layer_idx}]")
    for lang_name in LANGUAGE_NAMES:
        lang_top = top_tokens_df[top_tokens_df['language'] == lang_name].head(5)
        print(f"\n{lang_name.upper()}:")
        for _, row in lang_top.iterrows():
            detected = row['detected_language']
            match_marker = "✓" if detected == lang_name else "✗"
            print(f"  {row['rank']:3d}. {row['token_text']:20s} (sim={row['similarity']:.4f}) [{detected:10s}] {match_marker}")

    # Calculate match rates for CSV output (convert to percentage)
    match_rates_data = []
    for lang_name in LANGUAGE_NAMES:
        lang_tokens = top_tokens_df[top_tokens_df['language'] == lang_name]
        match_stats = calculate_language_match_rate(lang_tokens, lang_name)

        match_rates_data.append({
            'model': model_name,
            'layer': layer_idx,
            'language': lang_name,
            'match_rate_pct': round(match_stats['match_rate'] * 100, 2),  # Convert to percentage
            'match_count': match_stats['match_count'],
            'total_sampled': match_stats['total_tokens']
        })

    match_rates_df = pd.DataFrame(match_rates_data)

    return {
        'top_tokens': top_tokens_df,
        'distribution': distribution_df,
        'match_rates': match_rates_df,
        'similarity_matrix': similarity_matrix,
        'manifold_dimensions': manifold_results['dimension_estimates'],
        'manifold_variance': manifold_results['variance_explained'],
        'pca_objects': manifold_results['pca_objects']
    }


# ============================================================================
# Probe Checkpoint Loading (for Steering Experiments)
# ============================================================================

def load_probe_checkpoint(checkpoint_path, device=DEVICE):
    """
    Load a saved probe checkpoint

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load probe onto

    Returns:
        dict with keys:
        - 'probe': Reconstructed LinearProbe model
        - 'metadata': Dictionary with training info
        - 'weights': Raw classifier weights for steering

    Usage:
        >>> loaded = load_probe_checkpoint('./checkpoints/meta-llama_Llama-3.1-8B-Instruct/layer_20_probe.pt')
        >>> probe = loaded['probe']
        >>> weights = loaded['weights']
        >>> en_direction = weights['classifier_weight'][0]  # English steering vector
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct probe
    probe = LinearProbe(
        hidden_size=checkpoint['hidden_size'],
        num_classes=checkpoint['num_classes']
    )
    probe.load_state_dict(checkpoint['probe_state_dict'])
    probe = probe.to(device)
    probe.eval()

    # Extract metadata
    metadata = {
        'layer_idx': checkpoint['layer_idx'],
        'model_name': checkpoint['model_name'],
        'val_accuracy': checkpoint['val_accuracy'],
        'hidden_size': checkpoint['hidden_size'],
        'num_classes': checkpoint['num_classes'],
        'train_size': checkpoint.get('train_size', None),
        'val_size': checkpoint.get('val_size', None),
    }

    # Extract raw weights for steering
    weights = {
        'classifier_weight': checkpoint['classifier_weight'],  # (num_classes, hidden_size)
        'classifier_bias': checkpoint['classifier_bias'],
        'layernorm_weight': checkpoint['layernorm_weight'],
        'layernorm_bias': checkpoint['layernorm_bias'],
    }

    return {
        'probe': probe,
        'metadata': metadata,
        'weights': weights
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_model_experiment(model_name, train_dataset, val_dataset, num_classes):
    """Run experiment for single model"""
    print("\n" + "="*70)
    print(f"MODEL: {model_name}")
    print("="*70)

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")

    # Use bfloat16 for gpt-oss models, float16 for others
    dtype = torch.bfloat16 if "gpt-oss" in model_name else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=DEVICE,
        trust_remote_code=True
    )

    print(f"✓ Model loaded")

    # Get config
    config = getattr(model.config, 'text_config', model.config)
    num_layers = getattr(config, 'num_hidden_layers',
                        getattr(config, 'num_layers', None))
    if num_layers is None:
        raise ValueError(f"Could not find num_layers attribute in model config")

    hidden_size = getattr(config, 'hidden_size', None)
    if hidden_size is None:
        raise ValueError(f"Could not find hidden_size attribute in model config")

    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")

    # Extract features once
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)

    train_feats, train_labels = extract_features(model, tokenizer, train_dataset)
    val_feats, val_labels = extract_features(model, tokenizer, val_dataset)

    # Collect results
    all_accuracies = []
    all_distributions = []
    all_match_rates = []
    all_geometry_stats = []
    all_geometry_metrics = []
    all_entropy_stats = []
    all_top_predictions = []
    all_manifold_dimensions = []
    all_manifold_variance = []
    cosine_matrices = {}

    # Process each layer
    print("\n" + "="*70)
    print(f"ANALYZING ALL LAYERS ({num_layers} layers)")
    print("="*70)

    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*70}")

        # Prepare datasets
        train_dataset_feat = FeatureDataset(train_feats[layer_idx], train_labels)
        val_dataset_feat = FeatureDataset(val_feats[layer_idx], val_labels)

        train_loader = DataLoader(train_dataset_feat, batch_size=TRAIN_BS, shuffle=True)
        val_loader = DataLoader(val_dataset_feat, batch_size=128, shuffle=False)

        # Train probe
        print("\n[Training Linear Probe]")
        probe = LinearProbe(hidden_size, num_classes)
        probe, val_acc = train_probe(probe, train_loader, val_loader)
        print(f"  Val Accuracy: {val_acc:.4f}")

        # Store accuracy
        all_accuracies.append({
            'model': model_name,
            'layer': layer_idx,
            'val_accuracy': round(val_acc * 100, 2)  # Convert to percentage
        })

        # *** SAVE PROBE CHECKPOINT FOR STEERING EXPERIMENTS ***
        checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', model_name.replace('/', '_').replace(':', '_'))
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'layer_{layer_idx}_probe.pt')
        checkpoint_data = {
            # Full state dict for reloading probe
            'probe_state_dict': probe.state_dict(),

            # Raw classifier weights (for steering interventions)
            'classifier_weight': probe.classifier.weight.data.cpu().clone(),
            'classifier_bias': probe.classifier.bias.data.cpu().clone() if hasattr(probe.classifier, 'bias') and probe.classifier.bias is not None else None,

            # LayerNorm parameters (for proper normalization during steering)
            'layernorm_weight': probe.norm.weight.data.cpu().clone(),
            'layernorm_bias': probe.norm.bias.data.cpu().clone(),

            # Metadata for reconstruction
            'hidden_size': hidden_size,
            'num_classes': num_classes,
            'layer_idx': layer_idx,
            'model_name': model_name,
            'val_accuracy': val_acc,

            # Optional: training stats for analysis
            'train_size': len(train_labels),
            'val_size': len(val_labels),
        }

        torch.save(checkpoint_data, checkpoint_path)
        print(f"  ✓ Saved probe checkpoint: {checkpoint_path}")

        # Analyze probe geometry
        geometry_results = analyze_probe_geometry(probe, layer_idx, model_name)
        all_geometry_stats.append(pd.DataFrame([geometry_results['orthogonality_stats']]))
        all_geometry_metrics.append(pd.DataFrame(geometry_results['geometry_metrics']))
        cosine_matrices[layer_idx] = geometry_results['cosine_matrix_df']

        # Analyze token-language alignment
        analysis_results = analyze_token_language_alignment(
            model, tokenizer, probe, layer_idx, model_name
        )

        # Skip if NaN was detected
        if analysis_results is None:
            continue

        all_distributions.append(analysis_results['distribution'])
        all_match_rates.append(analysis_results['match_rates'])

        # Store manifold analysis results
        if 'manifold_dimensions' in analysis_results:
            all_manifold_dimensions.append(analysis_results['manifold_dimensions'])
            all_manifold_variance.append(analysis_results['manifold_variance'])

        # Generation validation (only for key layers to save time)
        key_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
        if layer_idx in key_layers:
            try:
                gen_results = generation_level_validation(
                    model, tokenizer, probe, layer_idx, model_name, num_samples=10
                )
                all_entropy_stats.append(gen_results['entropy_stats'])
                all_top_predictions.append(gen_results['top_predictions'])
            except Exception as e:
                print(f"  Warning: Generation validation failed for layer {layer_idx}: {str(e)}")

        # Cleanup
        del probe, train_loader, val_loader, train_dataset_feat, val_dataset_feat
        torch.cuda.empty_cache()

    # Cleanup features
    del train_feats, val_feats
    torch.cuda.empty_cache()

    # Combine results
    all_accuracies_df = pd.DataFrame(all_accuracies)
    all_distributions_df = pd.concat(all_distributions, ignore_index=True) if all_distributions else pd.DataFrame()
    all_match_rates_df = pd.concat(all_match_rates, ignore_index=True) if all_match_rates else pd.DataFrame()
    all_geometry_stats_df = pd.concat(all_geometry_stats, ignore_index=True) if all_geometry_stats else pd.DataFrame()
    all_geometry_metrics_df = pd.concat(all_geometry_metrics, ignore_index=True) if all_geometry_metrics else pd.DataFrame()
    all_entropy_stats_df = pd.concat(all_entropy_stats, ignore_index=True) if all_entropy_stats else pd.DataFrame()
    all_top_predictions_df = pd.concat(all_top_predictions, ignore_index=True) if all_top_predictions else pd.DataFrame()
    all_manifold_dimensions_df = pd.concat(all_manifold_dimensions, ignore_index=True) if all_manifold_dimensions else pd.DataFrame()
    all_manifold_variance_df = pd.concat(all_manifold_variance, ignore_index=True) if all_manifold_variance else pd.DataFrame()

    return (all_accuracies_df, all_distributions_df, all_match_rates_df,
            all_geometry_stats_df, all_geometry_metrics_df, cosine_matrices,
            all_entropy_stats_df, all_top_predictions_df,
            all_manifold_dimensions_df, all_manifold_variance_df,
            model, tokenizer)


def create_visualizations(distributions_df, model_name):
    """Create visualization plots"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    # Filter for this model
    model_dist = distributions_df[distributions_df['model'] == model_name]

    # 1. Language distribution across layers
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Vocab percentage
    pivot_pct = model_dist.pivot(index='layer', columns='language', values='vocab_percentage')
    pivot_pct.plot(kind='line', marker='o', ax=axes[0])
    axes[0].set_title(f'Vocab Percentage by Language Across Layers - {model_name}')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Vocabulary Percentage (%)')
    axes[0].legend(title='Language')
    axes[0].grid(True, alpha=0.3)

    # Average similarity
    pivot_sim = model_dist.pivot(index='layer', columns='language', values='avg_similarity')
    pivot_sim.plot(kind='line', marker='o', ax=axes[1])
    axes[1].set_title(f'Average Similarity by Language Across Layers - {model_name}')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average Similarity')
    axes[1].legend(title='Language')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with safe filename
    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    output_path = os.path.join(FIGURES_DIR, f'language_distribution_{safe_model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def create_geometry_visualizations(cosine_matrices, model_name):
    """Create heatmap visualizations of probe geometry"""
    print("\n" + "="*70)
    print("CREATING PROBE GEOMETRY VISUALIZATIONS")
    print("="*70)

    # Select key layers (early, middle, late)
    layers_to_plot = sorted(cosine_matrices.keys())
    num_layers = len(layers_to_plot)

    if num_layers == 0:
        print("No geometry data to visualize")
        return

    # Plot up to 6 layers in 2×3 grid
    plot_layers = layers_to_plot[::max(1, num_layers//6)][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, layer_idx in enumerate(plot_layers):
        cosine_df = cosine_matrices[layer_idx]

        sns.heatmap(
            cosine_df,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=axes[idx],
            cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[idx].set_title(f'Layer {layer_idx}')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    # Hide unused subplots
    for idx in range(len(plot_layers), 6):
        axes[idx].axis('off')

    plt.suptitle(f'Probe Geometry: Language Direction Cosine Similarity - {model_name}', fontsize=16)
    plt.tight_layout()

    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    output_path = os.path.join(FIGURES_DIR, f'probe_geometry_{safe_model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved geometry visualization: {output_path}")
    plt.close()


def create_generation_visualizations(entropy_stats_df, model_name):
    """Visualize entropy comparison across languages and layers"""
    print("\n" + "="*70)
    print("CREATING GENERATION VALIDATION VISUALIZATIONS")
    print("="*70)

    model_entropy = entropy_stats_df[entropy_stats_df['model'] == model_name]

    if len(model_entropy) == 0:
        print("No entropy data to visualize")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Entropy by language across layers
    for lang in LANGUAGE_NAMES:
        lang_data = model_entropy[model_entropy['language'] == lang]
        if len(lang_data) > 0:
            axes[0].plot(lang_data['layer'], lang_data['mean_entropy'],
                        marker='o', label=lang, linewidth=2)
            axes[0].fill_between(
                lang_data['layer'],
                lang_data['mean_entropy'] - lang_data['std_entropy'],
                lang_data['mean_entropy'] + lang_data['std_entropy'],
                alpha=0.2
            )

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Next-Token Entropy')
    axes[0].set_title(f'Next-Token Prediction Entropy - {model_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Entropy distribution (box plot)
    entropy_by_lang = []
    for lang in LANGUAGE_NAMES:
        lang_data = model_entropy[model_entropy['language'] == lang]
        if len(lang_data) > 0:
            entropy_by_lang.append(lang_data['mean_entropy'].values)

    if entropy_by_lang:
        axes[1].boxplot(entropy_by_lang, labels=LANGUAGE_NAMES)
        axes[1].set_ylabel('Mean Entropy')
        axes[1].set_xlabel('Language')
        axes[1].set_title('Entropy Distribution Across Layers')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    output_path = os.path.join(FIGURES_DIR, f'generation_entropy_{safe_model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved generation visualization: {output_path}")
    plt.close()


def create_manifold_visualizations(dimension_df, variance_df, model_name):
    """Visualize manifold dimension estimates"""
    print("\n" + "="*70)
    print("CREATING MANIFOLD DIMENSION VISUALIZATIONS")
    print("="*70)

    model_dim = dimension_df[dimension_df['model'] == model_name]
    model_var = variance_df[variance_df['model'] == model_name]

    if len(model_dim) == 0:
        print("No manifold data to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Intrinsic dimensions across layers
    for lang in LANGUAGE_NAMES:
        lang_data = model_dim[model_dim['language'] == lang]
        if len(lang_data) > 0:
            axes[0, 0].plot(lang_data['layer'], lang_data['dim_95pct'],
                           marker='o', label=lang, linewidth=2)

    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Intrinsic Dimension (95% variance)')
    axes[0, 0].set_title('Intrinsic Dimensionality Across Layers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Effective dimension comparison
    for lang in LANGUAGE_NAMES:
        lang_data = model_dim[model_dim['language'] == lang]
        if len(lang_data) > 0:
            axes[0, 1].plot(lang_data['layer'], lang_data['effective_dim'],
                           marker='o', label=lang, linewidth=2)

    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Effective Dimension')
    axes[0, 1].set_title('Effective Dimensionality (Entropy-based)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Scree plot for final layer
    if len(model_var) > 0:
        final_layer = model_var['layer'].max()
        final_var = model_var[model_var['layer'] == final_layer]

        for lang in LANGUAGE_NAMES:
            lang_var = final_var[final_var['language'] == lang]
            if len(lang_var) > 0:
                # Plot first 20 components
                lang_var_subset = lang_var[lang_var['component'] <= 20]
                axes[1, 0].plot(lang_var_subset['component'],
                               lang_var_subset['variance_ratio']*100,
                               marker='o', label=lang, linewidth=2)

        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Variance Explained (%)')
        axes[1, 0].set_title(f'Scree Plot - Layer {final_layer} (First 20 PCs)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cumulative variance
        for lang in LANGUAGE_NAMES:
            lang_var = final_var[final_var['language'] == lang]
            if len(lang_var) > 0:
                lang_var_subset = lang_var[lang_var['component'] <= 50]
                axes[1, 1].plot(lang_var_subset['component'],
                               lang_var_subset['cumulative_variance']*100,
                               marker='o', label=lang, linewidth=2, markersize=4)

        axes[1, 1].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=95, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance Explained (%)')
        axes[1, 1].set_title(f'Cumulative Variance - Layer {final_layer}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Manifold Dimension Analysis - {model_name}', fontsize=16)
    plt.tight_layout()

    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    output_path = os.path.join(FIGURES_DIR, f'manifold_dimensions_{safe_model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved manifold visualization: {output_path}")
    plt.close()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("TOKEN-LANGUAGE ALIGNMENT ANALYSIS")
    print("="*70)
    print(f"Models: {len(MODEL_LIST)}")
    for i, model in enumerate(MODEL_LIST, 1):
        print(f"  {i}. {model}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # Load data
    train_dataset, val_dataset, test_dataset, metadata = load_data()
    num_classes = metadata['num_classes']

    all_accuracies_results = []
    all_distribution_results = []
    all_match_rates_results = []
    all_geometry_stats_results = []
    all_geometry_metrics_results = []
    all_entropy_stats_results = []
    all_top_predictions_results = []
    all_manifold_dimensions_results = []
    all_manifold_variance_results = []

    for model_idx, model_name in enumerate(MODEL_LIST, 1):
        print("\n" + "#"*70)
        print(f"# MODEL {model_idx}/{len(MODEL_LIST)}: {model_name}")
        print("#"*70)

        try:
            (accuracies_df, distributions_df, match_rates_df,
             geometry_stats_df, geometry_metrics_df, cosine_matrices,
             entropy_stats_df, top_predictions_df,
             manifold_dimensions_df, manifold_variance_df,
             model, tokenizer) = run_single_model_experiment(
                model_name, train_dataset, val_dataset, num_classes
            )

            all_accuracies_results.append(accuracies_df)
            all_distribution_results.append(distributions_df)
            all_match_rates_results.append(match_rates_df)
            all_geometry_stats_results.append(geometry_stats_df)
            all_geometry_metrics_results.append(geometry_metrics_df)
            all_entropy_stats_results.append(entropy_stats_df)
            all_top_predictions_results.append(top_predictions_df)
            all_manifold_dimensions_results.append(manifold_dimensions_df)
            all_manifold_variance_results.append(manifold_variance_df)

            # Create visualizations
            create_visualizations(distributions_df, model_name)
            create_geometry_visualizations(cosine_matrices, model_name)
            create_generation_visualizations(entropy_stats_df, model_name)
            create_manifold_visualizations(manifold_dimensions_df, manifold_variance_df, model_name)

            # Cleanup model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"\n✗ Error with model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save final results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Combine results from all models
    final_accuracies_df = pd.concat(all_accuracies_results, ignore_index=True)
    final_distributions_df = pd.concat(all_distribution_results, ignore_index=True)
    final_match_rates_df = pd.concat(all_match_rates_results, ignore_index=True)

    # File 1: Probe accuracies (model, layer, val_accuracy)
    accuracies_file = os.path.join(OUTPUT_DIR, 'probe_accuracies.csv')
    final_accuracies_df.to_csv(accuracies_file, index=False)
    print(f"✓ Saved probe accuracies: {accuracies_file}")

    # File 2: Token-language alignment (merge distributions and match_rates)
    # Merge on model, layer, language
    alignment_df = pd.merge(
        final_distributions_df,
        final_match_rates_df,
        on=['model', 'layer', 'language'],
        how='left'
    )

    # Reorder columns for better readability
    alignment_df = alignment_df[[
        'model', 'layer', 'language',
        'token_count', 'vocab_percentage', 'avg_similarity',
        'match_rate_pct', 'match_count', 'total_sampled'
    ]]

    alignment_file = os.path.join(OUTPUT_DIR, 'token_language_alignment.csv')
    alignment_df.to_csv(alignment_file, index=False)
    print(f"✓ Saved token-language alignment: {alignment_file}")

    # File 3: Probe geometry stats
    if all_geometry_stats_results:
        final_geometry_stats_df = pd.concat(all_geometry_stats_results, ignore_index=True)
        geometry_stats_file = os.path.join(OUTPUT_DIR, 'probe_geometry_stats.csv')
        final_geometry_stats_df.to_csv(geometry_stats_file, index=False)
        print(f"✓ Saved probe geometry stats: {geometry_stats_file}")

    # File 4: Probe geometry metrics
    if all_geometry_metrics_results:
        final_geometry_metrics_df = pd.concat(all_geometry_metrics_results, ignore_index=True)
        geometry_metrics_file = os.path.join(OUTPUT_DIR, 'probe_geometry_metrics.csv')
        final_geometry_metrics_df.to_csv(geometry_metrics_file, index=False)
        print(f"✓ Saved probe geometry metrics: {geometry_metrics_file}")

    # File 5: Generation entropy stats
    if all_entropy_stats_results:
        final_entropy_stats_df = pd.concat(all_entropy_stats_results, ignore_index=True)
        entropy_file = os.path.join(OUTPUT_DIR, 'generation_entropy_stats.csv')
        final_entropy_stats_df.to_csv(entropy_file, index=False)
        print(f"✓ Saved generation entropy stats: {entropy_file}")

    # File 6: Generation top predictions
    if all_top_predictions_results:
        final_top_predictions_df = pd.concat(all_top_predictions_results, ignore_index=True)
        predictions_file = os.path.join(OUTPUT_DIR, 'generation_top_predictions.csv')
        final_top_predictions_df.to_csv(predictions_file, index=False)
        print(f"✓ Saved generation top predictions: {predictions_file}")

    # File 7: Manifold dimension estimates
    if all_manifold_dimensions_results:
        final_manifold_dim_df = pd.concat(all_manifold_dimensions_results, ignore_index=True)
        manifold_dim_file = os.path.join(OUTPUT_DIR, 'manifold_dimension_estimates.csv')
        final_manifold_dim_df.to_csv(manifold_dim_file, index=False)
        print(f"✓ Saved manifold dimension estimates: {manifold_dim_file}")

    # File 8: Manifold variance explained
    if all_manifold_variance_results:
        final_manifold_var_df = pd.concat(all_manifold_variance_results, ignore_index=True)
        manifold_var_file = os.path.join(OUTPUT_DIR, 'manifold_variance_explained.csv')
        final_manifold_var_df.to_csv(manifold_var_file, index=False)
        print(f"✓ Saved manifold variance explained: {manifold_var_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for model_name in MODEL_LIST:
        model_dist = final_distributions_df[final_distributions_df['model'] == model_name]
        if len(model_dist) == 0:
            continue

        print(f"\n[{model_name}]")

        # Average distribution across layers
        avg_dist = model_dist.groupby('language')['vocab_percentage'].mean()
        print("  Average language distribution across all layers:")
        for lang, pct in avg_dist.items():
            print(f"    {lang}: {pct:.2f}%")

        # Layer with highest differentiation
        layer_std = model_dist.groupby('layer')['vocab_percentage'].std()
        best_layer = layer_std.idxmax()
        print(f"  Layer with highest language differentiation: {best_layer}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
