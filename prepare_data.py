"""
Data Preparation Script for Multilingual Language Classification

This script downloads and prepares the XNLI dataset for the experiments.
It creates a pickle file with train/val/test splits for 5 languages.

Usage:
    python prepare_data.py --output_dir ./data --languages en es zh fr de

Output:
    ./data/multilingual_xnli_5lang.pkl
"""

import os
import argparse
import pickle
import numpy as np
from datasets import load_dataset
from collections import defaultdict


def load_xnli_data(languages=['en', 'es', 'zh', 'fr', 'de'],
                   train_samples_per_lang=5000,
                   val_samples_per_lang=2500):
    """
    Load XNLI dataset and prepare for language classification

    Args:
        languages: List of language codes
        train_samples_per_lang: Number of training samples per language
        val_samples_per_lang: Number of validation samples per language

    Returns:
        Dictionary with train/val/test data
    """
    print("="*70)
    print("LOADING XNLI DATASET")
    print("="*70)
    print(f"Languages: {languages}")
    print(f"Train samples per language: {train_samples_per_lang}")
    print(f"Val samples per language: {val_samples_per_lang}")

    # Language mapping (XNLI uses different codes)
    lang_map = {
        'en': 'en',
        'es': 'es',
        'zh': 'zh',  # XNLI uses 'zh' for Chinese
        'fr': 'fr',
        'de': 'de'
    }

    # Create language to index mapping
    lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}

    # Storage
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)

    # Load XNLI dataset
    print("\nDownloading XNLI from Hugging Face...")

    for lang_code in languages:
        xnli_lang = lang_map[lang_code]

        print(f"\nProcessing {lang_code} ({xnli_lang})...")

        # Load train set (use premise as text for language classification)
        try:
            dataset = load_dataset('xnli', xnli_lang)

            # Training data
            train_dataset = dataset['train']
            train_texts = [item['premise'] for item in train_dataset]

            # Limit samples
            if len(train_texts) > train_samples_per_lang:
                np.random.seed(42)
                indices = np.random.choice(len(train_texts), train_samples_per_lang, replace=False)
                train_texts = [train_texts[i] for i in indices]

            train_data['texts'].extend(train_texts)
            train_data['labels'].extend([lang_to_idx[lang_code]] * len(train_texts))

            print(f"  Train: {len(train_texts)} samples")

            # Validation data
            val_dataset = dataset['validation']
            val_texts = [item['premise'] for item in val_dataset]

            if len(val_texts) > val_samples_per_lang:
                indices = np.random.choice(len(val_texts), val_samples_per_lang, replace=False)
                val_texts = [val_texts[i] for i in indices]

            val_data['texts'].extend(val_texts)
            val_data['labels'].extend([lang_to_idx[lang_code]] * len(val_texts))

            print(f"  Val: {len(val_texts)} samples")

            # Test data (optional, use validation if not available)
            if 'test' in dataset:
                test_dataset = dataset['test']
                test_texts = [item['premise'] for item in test_dataset]

                if len(test_texts) > val_samples_per_lang:
                    indices = np.random.choice(len(test_texts), val_samples_per_lang, replace=False)
                    test_texts = [test_texts[i] for i in indices]

                test_data['texts'].extend(test_texts)
                test_data['labels'].extend([lang_to_idx[lang_code]] * len(test_texts))

                print(f"  Test: {len(test_texts)} samples")

        except Exception as e:
            print(f"  Error loading {lang_code}: {str(e)}")
            continue

    # Convert to numpy arrays
    train_data['labels'] = np.array(train_data['labels'])
    val_data['labels'] = np.array(val_data['labels'])

    if test_data['texts']:
        test_data['labels'] = np.array(test_data['labels'])
    else:
        # Use validation as test if test set not available
        print("\nNo test set available, using validation set as test")
        test_data = val_data.copy()

    # Create final data structure
    data = {
        'metadata': {
            'languages': languages,
            'num_classes': len(languages),
            'total_train': len(train_data['texts']),
            'total_val': len(val_data['texts']),
            'total_test': len(test_data['texts']),
            'lang_to_idx': lang_to_idx
        },
        'train': {
            'texts': train_data['texts'],
            'labels': train_data['labels']
        },
        'val': {
            'texts': val_data['texts'],
            'labels': val_data['labels']
        },
        'test': {
            'texts': test_data['texts'],
            'labels': test_data['labels']
        }
    }

    return data


def main():
    parser = argparse.ArgumentParser(description='Prepare XNLI data for language classification')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for processed data')
    parser.add_argument('--languages', nargs='+', default=['en', 'es', 'zh', 'fr', 'de'],
                       help='List of language codes')
    parser.add_argument('--train_samples', type=int, default=5000,
                       help='Number of training samples per language')
    parser.add_argument('--val_samples', type=int, default=2500,
                       help='Number of validation samples per language')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare data
    data = load_xnli_data(
        languages=args.languages,
        train_samples_per_lang=args.train_samples,
        val_samples_per_lang=args.val_samples
    )

    # Save to pickle
    output_path = os.path.join(args.output_dir, 'multilingual_xnli_5lang.pkl')

    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    print(f"Output: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Data saved successfully")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Languages: {data['metadata']['languages']}")
    print(f"Number of classes: {data['metadata']['num_classes']}")
    print(f"Train samples: {data['metadata']['total_train']}")
    print(f"Val samples: {data['metadata']['total_val']}")
    print(f"Test samples: {data['metadata']['total_test']}")

    # Per-language breakdown
    print("\nPer-language distribution:")
    for lang_idx, lang in enumerate(data['metadata']['languages']):
        train_count = (data['train']['labels'] == lang_idx).sum()
        val_count = (data['val']['labels'] == lang_idx).sum()
        test_count = (data['test']['labels'] == lang_idx).sum()
        print(f"  {lang}: Train={train_count}, Val={val_count}, Test={test_count}")

    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nYou can now use this data with the analysis scripts.")
    print(f"Update DATA_PATH in the scripts to: {output_path}")


if __name__ == '__main__':
    main()
