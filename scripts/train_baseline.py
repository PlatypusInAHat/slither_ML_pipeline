"""
Main training script for Optimized-CodeBERT model.
Loads data, initializes model, and runs training.
"""

import argparse
import json
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.dataset import VulnerabilityDataset, split_dataset
from src.ml.models import create_model
from src.ml.train import Trainer, evaluate_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function."""
    
    print("="*80)
    print("Lightning Cat - Optimized-CodeBERT Training")
    print("="*80)
    print()
    
    # Load configuration
    config_path = args.config or "configs/train.yaml"
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Prepare data paths
    data_dir = Path(args.data_dir or config['paths']['data_dir'])
    
    # Check if data is already split
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    test_file = data_dir / "test.jsonl"
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        print("Split files not found. Looking for combined data file...")
        
        # Look for a combined JSONL file
        combined_file = None
        for candidate in data_dir.glob("*.jsonl"):
            if candidate.name not in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                combined_file = candidate
                break
        
        if combined_file is None:
            raise FileNotFoundError(
                f"No data files found in {data_dir}. "
                "Please provide either split files (train/val/test.jsonl) "
                "or a combined JSONL file."
            )
        
        print(f"Found combined file: {combined_file}")
        print("Splitting dataset...")
        
        train_file, val_file, test_file = split_dataset(
            input_file=str(combined_file),
            output_dir=str(data_dir),
            splits=tuple(config.get('train_val_test', [0.8, 0.1, 0.1])),
            seed=seed
        )
        print()
    
    # Load datasets
    print("Loading datasets...")
    max_len = config.get('max_len', 512)
    model_name = config['model']['name']
    
    train_dataset = VulnerabilityDataset(
        data_path=str(train_file),
        tokenizer_name=model_name,
        max_length=max_len,
        split="train"
    )
    
    val_dataset = VulnerabilityDataset(
        data_path=str(val_file),
        tokenizer_name=model_name,
        max_length=max_len,
        split="validation"
    )
    
    test_dataset = VulnerabilityDataset(
        data_path=str(test_file),
        tokenizer_name=model_name,
        max_length=max_len,
        split="test"
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print()
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        print("Calculating class weights for imbalanced data...")
        class_weights = VulnerabilityDataset.get_label_weights(str(train_file))
        print(f"Class weights: {class_weights}")
        print()
    
    # Create model
    print("Initializing model...")
    model = create_model(config, class_weights=class_weights)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {config['model']['name']}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train model
    if not args.skip_training:
        training_results = trainer.train()
        
        # Save training results
        output_dir = Path(config['paths']['output_dir'])
        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        print(f"Training results saved to: {results_file}")
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    best_checkpoint_path = Path(config['paths']['output_dir']) / 'checkpoint_best.pt'
    
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    else:
        print("Warning: Best checkpoint not found. Using current model state.")
    
    # Evaluate on test set
    label_names = ['reentrancy', 'timestamp_dependency', 'unchecked_call', 'tx_origin_misuse', 'safe']
    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        label_names=label_names
    )
    
    # Save test results
    output_dir = Path(config['paths']['output_dir'])
    test_results_file = output_dir / 'test_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    test_results_json = {
        'accuracy': float(test_results['accuracy']),
        'precision': float(test_results['precision']),
        'recall': float(test_results['recall']),
        'f1': float(test_results['f1']),
        'precision_per_class': test_results['precision_per_class'].tolist(),
        'recall_per_class': test_results['recall_per_class'].tolist(),
        'f1_per_class': test_results['f1_per_class'].tolist(),
        'support': test_results['support'].tolist(),
        'confusion_matrix': test_results['confusion_matrix'].tolist(),
        'classification_report': test_results['classification_report']
    }
    
    with open(test_results_file, 'w') as f:
        json.dump(test_results_json, f, indent=2)
    
    print(f"\nTest results saved to: {test_results_file}")
    
    # Save final model
    final_model_path = output_dir / 'model_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_results': test_results_json
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    
    print("\n" + "="*80)
    print("Training and evaluation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Optimized-CodeBERT for vulnerability detection"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train.yaml',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing training data (overrides config)'
    )
    
    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        help='Use class weights to handle imbalanced data'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only evaluate existing model'
    )
    
    args = parser.parse_args()
    
    main(args)
