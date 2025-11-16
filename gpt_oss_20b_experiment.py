"""
GPT-OSS-20B 전용 실험 스크립트

단일 모델 (openai/gpt-oss-20b)에 대해 5개 언어 분류 실험
Multi-seed (5 seeds) for statistical robustness

GPU 3번만 사용

Usage:
    CUDA_VISIBLE_DEVICES=3 python gpt_oss_20b_experiment.py

Environment:
    conda activate only_for_VLLM
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
import gc
import random

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"

# Random seeds for robustness
# SEEDS = [42, 123, 456, 789, 2024]  # Original
SEEDS = [42, 123, 456, 789, 2024]  # All 5 seeds for statistical robustness

DATA_PATH = "/home/dilab05/work_directory/김재성/지티어트랙션_시험/ICLR/Data/multilingual_xnli_5lang.pkl"
OUTPUT_DIR = "./gpt_oss_20b_results"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "all_seeds_results.csv")
AGGREGATED_FILE = os.path.join(OUTPUT_DIR, "aggregated_results_mean_std.csv")

# Hyperparameters
EXTRACT_BS = 8
TRAIN_BS = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 3
MAX_LENGTH = 256

# GPU 설정 (GPU 3번만 사용)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"GPU 3 selected for {MODEL_NAME}")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================================
# Seed Setting
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n{'='*70}")
    print(f"SEED SET TO: {seed}")
    print(f"{'='*70}")


# ============================================================================
# Dataset
# ============================================================================

class LanguageDataset(Dataset):
    """언어 분류 데이터셋"""

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


# ============================================================================
# Probe Models
# ============================================================================

class LinearProbe(nn.Module):
    """선형 probe: LayerNorm + Linear"""

    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.norm(x)
        return self.classifier(x)


class MLPProbe(nn.Module):
    """MLP probe: LayerNorm + MLP (hidden layer 512)"""

    def __init__(self, hidden_size, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """데이터 로드"""
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

    # Dataset 생성
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
    """모든 layer에서 hidden states 추출 (Chat template 적용)"""
    print("\nExtracting features from all layers...")
    print("NOTE: Applying chat template to input texts")

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 각 layer별로 features 저장
    config = getattr(model.config, 'text_config', model.config)
    num_layers = getattr(config, 'num_hidden_layers',
                        getattr(config, 'num_layers', None))
    if num_layers is None:
        raise ValueError(f"Could not find num_layers attribute in model config")

    print(f"Number of layers: {num_layers}")

    all_features = {layer_idx: [] for layer_idx in range(num_layers)}
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Feature extraction"):
            texts = batch['text']
            labels = batch['label'].numpy()

            # Apply chat template to each text
            formatted_texts = []
            for text in texts:
                messages = [{"role": "user", "content": text}]
                try:
                    formatted_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    formatted_texts.append(formatted_text)
                except Exception as e:
                    # If chat template fails, use raw text
                    print(f"Warning: Chat template failed, using raw text. Error: {e}")
                    formatted_texts.append(text)

            # Tokenize with chat template applied
            encoded = tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(DEVICE)

            # Forward pass with hidden states
            outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Last token pooling
            for layer_idx in range(num_layers):
                layer_hidden = hidden_states[layer_idx]
                last_token_hidden = layer_hidden[:, -1, :]
                # Convert BFloat16 to float32 before numpy conversion
                all_features[layer_idx].append(last_token_hidden.float().cpu().numpy())

            all_labels.append(labels)

    # Concatenate
    for layer_idx in range(num_layers):
        all_features[layer_idx] = np.vstack(all_features[layer_idx])

    all_labels = np.concatenate(all_labels)

    print(f"✓ Extracted features from {num_layers} layers")
    print(f"  Feature shape per layer: {all_features[0].shape}")
    print(f"  Labels shape: {all_labels.shape}")

    return all_features, all_labels


def extract_features_no_save(model, tokenizer, train_dataset, val_dataset, test_dataset):
    """Train/Val/Test의 모든 layer features 추출"""
    print("\n" + "="*70)
    print("FEATURE EXTRACTION (NO SAVE)")
    print("="*70)

    # Extract
    print("\nExtracting train features...")
    train_features, train_labels = extract_features(model, tokenizer, train_dataset)

    print("\nExtracting val features...")
    val_features, val_labels = extract_features(model, tokenizer, val_dataset)

    print("\nExtracting test features...")
    test_features, test_labels = extract_features(model, tokenizer, test_dataset)

    print("✓ Feature extraction complete (not saved)")

    return train_features, val_features, test_features, train_labels, val_labels, test_labels


# ============================================================================
# Probe Training
# ============================================================================

class FeatureDataset(Dataset):
    """Feature vector 데이터셋"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_probe(probe, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Probe 학습"""
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


def evaluate_probe(probe, test_loader):
    """Probe 평가 (전체 + 언어별)"""
    probe.eval()
    probe = probe.to(DEVICE)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            logits = probe(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall_acc = accuracy_score(all_labels, all_preds)

    # Per-language accuracy
    language_names = ['english', 'spanish', 'chinese', 'french', 'german']
    per_language_acc = {}

    for lang_idx, lang_name in enumerate(language_names):
        lang_mask = (all_labels == lang_idx)
        if lang_mask.sum() > 0:
            lang_acc = accuracy_score(all_labels[lang_mask], all_preds[lang_mask])
            per_language_acc[lang_name] = lang_acc
        else:
            per_language_acc[lang_name] = 0.0

    return overall_acc, per_language_acc, all_preds, all_labels


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment_for_layer(layer_idx, train_feats, val_feats, test_feats,
                             train_labels, val_labels, test_labels,
                             hidden_size, num_classes, model_name, seed):
    """특정 layer에 대한 Linear + MLP probe 실험"""
    print(f"\n{'='*70}")
    print(f"LAYER {layer_idx} | SEED {seed}")
    print(f"{'='*70}")

    # Prepare datasets
    train_dataset = FeatureDataset(train_feats[layer_idx], train_labels)
    val_dataset = FeatureDataset(val_feats[layer_idx], val_labels)
    test_dataset = FeatureDataset(test_feats[layer_idx], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Linear Probe
    print("\n[Linear Probe]")
    linear_probe = LinearProbe(hidden_size, num_classes)
    linear_probe, linear_val_acc_overall = train_probe(linear_probe, train_loader, val_loader)
    linear_test_acc_overall, linear_test_per_lang, _, _ = evaluate_probe(linear_probe, test_loader)
    linear_val_acc_overall_check, linear_val_per_lang, _, _ = evaluate_probe(linear_probe, val_loader)

    print(f"  Val Acc: {linear_val_acc_overall:.4f}, Test Acc: {linear_test_acc_overall:.4f}")

    # MLP Probe
    print("\n[MLP Probe]")
    mlp_probe = MLPProbe(hidden_size, num_classes)
    mlp_probe, mlp_val_acc_overall = train_probe(mlp_probe, train_loader, val_loader)
    mlp_test_acc_overall, mlp_test_per_lang, _, _ = evaluate_probe(mlp_probe, test_loader)
    mlp_val_acc_overall_check, mlp_val_per_lang, _, _ = evaluate_probe(mlp_probe, val_loader)

    print(f"  Val Acc: {mlp_val_acc_overall:.4f}, Test Acc: {mlp_test_acc_overall:.4f}")

    # Gap
    gap_overall = mlp_test_acc_overall - linear_test_acc_overall
    print(f"\n  MLP - Linear Gap: {gap_overall:+.4f} ({gap_overall*100:+.2f}%p)")

    # Create results list
    results_list = []

    # Overall result
    results_list.append({
        'seed': seed,
        'model': model_name,
        'layer': layer_idx,
        'language': 'overall',
        'linear_val_acc': linear_val_acc_overall,
        'linear_test_acc': linear_test_acc_overall,
        'mlp_val_acc': mlp_val_acc_overall,
        'mlp_test_acc': mlp_test_acc_overall,
        'mlp_linear_gap': gap_overall
    })

    # Per-language results
    for lang in ['english', 'spanish', 'chinese', 'french', 'german']:
        linear_val = linear_val_per_lang[lang]
        linear_test = linear_test_per_lang[lang]
        mlp_val = mlp_val_per_lang[lang]
        mlp_test = mlp_test_per_lang[lang]
        gap_lang = mlp_test - linear_test

        results_list.append({
            'seed': seed,
            'model': model_name,
            'layer': layer_idx,
            'language': lang,
            'linear_val_acc': linear_val,
            'linear_test_acc': linear_test,
            'mlp_val_acc': mlp_val,
            'mlp_test_acc': mlp_test,
            'mlp_linear_gap': gap_lang
        })

    # Cleanup
    del linear_probe, mlp_probe, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return results_list


def run_single_seed_experiment(model_name, train_dataset, val_dataset, test_dataset, num_classes, seed):
    """단일 seed에 대한 전체 실험 수행"""
    print("\n" + "="*70)
    print(f"MODEL: {model_name} | SEED: {seed}")
    print("="*70)

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad_token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # GPT-OSS uses BFloat16
        device_map="auto",
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

    all_layers = list(range(num_layers))

    # Extract features
    train_feats, val_feats, test_feats, train_labels, val_labels, test_labels = \
        extract_features_no_save(model, tokenizer, train_dataset, val_dataset, test_dataset)

    # Free model memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Run experiments for all layers
    print("\n" + "="*70)
    print(f"TRAINING PROBES FOR ALL LAYERS ({num_layers} layers)")
    print("="*70)

    model_results = []

    for layer_idx in all_layers:
        results_list = run_experiment_for_layer(
            layer_idx,
            train_feats, val_feats, test_feats,
            train_labels, val_labels, test_labels,
            hidden_size, num_classes, model_name, seed
        )
        model_results.extend(results_list)

    # Cleanup features
    del train_feats, val_feats, test_feats
    torch.cuda.empty_cache()
    gc.collect()

    return model_results


def aggregate_results(df):
    """Compute mean ± std across seeds"""
    print("\n" + "="*70)
    print("AGGREGATING RESULTS (MEAN ± STD)")
    print("="*70)

    # Group by model, layer, language
    grouped = df.groupby(['model', 'layer', 'language'])

    agg_results = []

    for (model, layer, language), group in grouped:
        result = {
            'model': model,
            'layer': layer,
            'language': language,
            'num_seeds': len(group),
        }

        # Compute mean and std for each metric
        for col in ['linear_val_acc', 'linear_test_acc', 'mlp_val_acc', 'mlp_test_acc', 'mlp_linear_gap']:
            mean_val = group[col].mean()
            std_val = group[col].std()
            result[f'{col}_mean'] = mean_val
            result[f'{col}_std'] = std_val
            result[f'{col}_mean_std'] = f"{mean_val:.4f} ± {std_val:.4f}"

        agg_results.append(result)

    return pd.DataFrame(agg_results)


def main():
    """메인 함수 - GPT-OSS-20B Multi-seed 실험"""
    print("\n" + "="*70)
    print("GPT-OSS-20B LANGUAGE CLASSIFICATION - MULTI-SEED EXPERIMENT")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} runs)")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # Load data once
    train_dataset, val_dataset, test_dataset, metadata = load_data()
    num_classes = metadata['num_classes']

    # Storage for all results
    all_results = []

    # Load existing results if file exists
    if os.path.exists(RESULTS_FILE):
        try:
            print(f"\n✓ Loading existing results from {RESULTS_FILE}")
            existing_df = pd.read_csv(RESULTS_FILE)
            all_results = existing_df.to_dict('records')
            print(f"  Loaded {len(all_results)} existing rows")

            # Get completed seeds
            completed_seeds = set(existing_df['seed'].unique())
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print(f"  Warning: {RESULTS_FILE} is empty or corrupted, starting fresh")
            completed_seeds = set()
    else:
        completed_seeds = set()

    initial_count = len(all_results)

    # Loop over seeds
    for seed in SEEDS:
        # Skip if already completed
        if seed in completed_seeds:
            print(f"\n✓ Skipping seed {seed} (already completed)")
            continue

        print("\n" + "#"*70)
        print(f"# SEED {seed} | MODEL: {MODEL_NAME}")
        print("#"*70)

        try:
            # Set seed before running experiment
            set_seed(seed)

            # Run experiment
            model_results = run_single_seed_experiment(
                MODEL_NAME, train_dataset, val_dataset, test_dataset, num_classes, seed
            )
            all_results.extend(model_results)

            # Save intermediate results after each seed
            df_temp = pd.DataFrame(all_results)
            df_temp.to_csv(RESULTS_FILE, index=False)
            print(f"\n✓ Intermediate results saved to {RESULTS_FILE}")

        except Exception as e:
            print(f"\n✗ Error with seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Final save of raw results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✓ All raw results saved to {RESULTS_FILE}")

    # Aggregate results (mean ± std)
    df_agg = aggregate_results(df)
    df_agg.to_csv(AGGREGATED_FILE, index=False)
    print(f"✓ Aggregated results saved to {AGGREGATED_FILE}")

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total runs completed: {len(all_results)} rows")
    print(f"New runs: {len(all_results) - initial_count} rows")
    print(f"Seeds used: {SEEDS}")

    # Show sample of aggregated results
    print("\n" + "="*70)
    print("SAMPLE AGGREGATED RESULTS (Overall only)")
    print("="*70)
    sample = df_agg[df_agg['language'] == 'overall'].head(10)
    print(sample[['layer', 'linear_test_acc_mean_std', 'mlp_test_acc_mean_std', 'mlp_linear_gap_mean_std']].to_string(index=False))

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  Raw: {RESULTS_FILE}")
    print(f"  Aggregated: {AGGREGATED_FILE}")


if __name__ == '__main__':
    main()
