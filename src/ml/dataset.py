"""
Dataset module for vulnerability detection using CodeBERT.
Loads JSONL data and prepares it for training.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import yaml


class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset for smart contract vulnerability detection.
    
    Loads function-level code snippets with vulnerability labels and tokenizes
    them using CodeBERT tokenizer.
    """
    
    # Label mapping: vulnerability type -> integer
    LABEL_MAP = {
        "reentrancy": 0,
        "timestamp_dependency": 1,
        "unchecked_call": 2,
        "tx_origin_misuse": 3,
        "safe": 4
    }
    
    # Reverse mapping for decoding predictions
    ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSONL file containing the data
            tokenizer_name: Name of the pre-trained tokenizer
            max_length: Maximum sequence length for tokenization
            split: Dataset split name (for logging)
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.split = split
        
        # Initialize CodeBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.samples = self._load_data()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_label_distribution()
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        samples = []
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        # Validate required fields
                        if 'code' in sample and 'label' in sample:
                            # Normalize label
                            label = sample['label'].lower()
                            if label in self.LABEL_MAP:
                                samples.append({
                                    'code': sample['code'],
                                    'label': label,
                                    'orig_id': sample.get('orig_id', ''),
                                    'func_name': sample.get('func_name', '')
                                })
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def _print_label_distribution(self):
        """Print distribution of labels in the dataset."""
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n{self.split} Label Distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(self.samples)) * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
        print()
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Label ID
        """
        sample = self.samples[idx]
        
        # Tokenize the code
        encoding = self.tokenizer(
            sample['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label ID
        label_id = self.LABEL_MAP[sample['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
    
    @staticmethod
    def get_num_labels() -> int:
        """Return the number of labels."""
        return len(VulnerabilityDataset.LABEL_MAP)
    
    @staticmethod
    def get_label_weights(data_path: str) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            data_path: Path to JSONL file
            
        Returns:
            Tensor of class weights
        """
        label_counts = {label: 0 for label in VulnerabilityDataset.LABEL_MAP.keys()}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        label = sample.get('label', '').lower()
                        if label in label_counts:
                            label_counts[label] += 1
                    except json.JSONDecodeError:
                        continue
        
        total = sum(label_counts.values())
        weights = []
        
        for label in sorted(VulnerabilityDataset.LABEL_MAP.keys(), 
                           key=lambda x: VulnerabilityDataset.LABEL_MAP[x]):
            count = label_counts[label]
            # Inverse frequency weighting
            weight = total / (len(label_counts) * count) if count > 0 else 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def split_dataset(
    input_file: str,
    output_dir: str,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split a JSONL dataset into train/val/test sets.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save split files
        splits: Tuple of (train, val, test) ratios
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    random.seed(seed)
    
    # Load all samples
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Calculate split indices
    total = len(samples)
    train_end = int(total * splits[0])
    val_end = train_end + int(total * splits[1])
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"
    test_file = output_path / "test.jsonl"
    
    for file_path, split_samples in [
        (train_file, train_samples),
        (val_file, val_samples),
        (test_file, test_samples)
    ]:
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                f.write(json.dumps(sample) + '\n')
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_samples)} samples -> {train_file}")
    print(f"  Val: {len(val_samples)} samples -> {val_file}")
    print(f"  Test: {len(test_samples)} samples -> {test_file}")
    
    return str(train_file), str(val_file), str(test_file)


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        dataset = VulnerabilityDataset(data_path, split="test")
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of labels: {dataset.get_num_labels()}")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample shape:")
            print(f"  input_ids: {sample['input_ids'].shape}")
            print(f"  attention_mask: {sample['attention_mask'].shape}")
            print(f"  labels: {sample['labels'].shape}")
    else:
        print("Usage: python dataset.py <path_to_jsonl>")
