# Optimized-CodeBERT Model - Usage Guide

This guide explains how to use the Optimized-CodeBERT model for smart contract vulnerability detection based on the Lightning Cat framework.

## Overview

The Optimized-CodeBERT model detects 4 types of vulnerabilities in Solidity smart contracts:
1. **Reentrancy**: Allows attackers to recursively call functions before state updates
2. **Timestamp Dependency**: Relies on block timestamps that can be manipulated by miners
3. **Unchecked Call**: Low-level calls without proper exception handling
4. **tx.origin Misuse**: Uses `tx.origin` instead of `msg.sender` for authentication

Additionally, the model classifies **safe** code without vulnerabilities.

## Model Architecture

- **Base Model**: Pre-trained CodeBERT (`microsoft/codebert-base`)
- **Encoder**: RoBERTa-based Transformer with multi-head self-attention
- **Classification Head**: 
  - Linear layer: 768 → 256 (ReLU activation)
  - Dropout layer (0.1)
  - Linear layer: 256 → 5 (output logits)
- **Parameters**: ~125M total, all trainable

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation:
```bash
python -c "from src.ml.models import OptimizedCodeBERT; print('✓ Installation successful')"
```

## Data Preparation

### Input Format

The model expects JSONL files with the following format:

```json
{"code": "function withdraw() public { ... }", "label": "reentrancy", "orig_id": "123", "func_name": "withdraw"}
{"code": "function transfer() public { ... }", "label": "safe", "orig_id": "124", "func_name": "transfer"}
```

**Required fields:**
- `code`: Solidity function source code (string)
- `label`: One of `reentrancy`, `timestamp_dependency`, `unchecked_call`, `tx_origin_misuse`, or `safe`

**Optional fields:**
- `orig_id`: Original contract ID
- `func_name`: Function name

### Data Preprocessing

The existing pipeline scripts handle data extraction:

1. **Extract contracts from HuggingFace dataset:**
```bash
python scripts/run_hf_pipeline.py
```

2. **Add safe labels using Slither:**
```bash
python scripts/add_safe_with_slither.py
```

This will create JSONL files in `data/processed/`.

### Dataset Splitting

The training script automatically splits data into train/val/test (80/10/10) if split files don't exist:

```bash
# Manual splitting (optional)
python -c "from src.ml.dataset import split_dataset; split_dataset('data/processed/all.jsonl', 'data/processed')"
```

## Training

### Basic Training

Train with default configuration:

```bash
python scripts/train_baseline.py
```

### Custom Configuration

Modify `configs/train.yaml` or create a new config file:

```yaml
seed: 42
max_len: 512
train_val_test: [0.8, 0.1, 0.1]

model:
  name: "microsoft/codebert-base"
  num_labels: 5
  dropout: 0.1
  classifier_hidden_dim: 256

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 500
  weight_decay: 0.01
```

Train with custom config:

```bash
python scripts/train_baseline.py --config configs/my_config.yaml
```

### Handling Imbalanced Data

Use class weights to handle imbalanced datasets:

```bash
python scripts/train_baseline.py --use-class-weights
```

### Training Arguments

```bash
python scripts/train_baseline.py \
  --config configs/train.yaml \
  --data-dir data/processed \
  --use-class-weights
```

**Arguments:**
- `--config`: Path to configuration file (default: `configs/train.yaml`)
- `--data-dir`: Directory containing JSONL data files
- `--use-class-weights`: Enable weighted loss for imbalanced data
- `--skip-training`: Skip training and only evaluate existing model

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir logs
```

Open browser to `http://localhost:6006` to see:
- Training/validation loss curves
- Accuracy, precision, recall, F1-score
- Learning rate schedule

### Console Output

Training progress is printed to console:

```
Epoch 1/10
------------------------------------------------------------
Epoch 1 | Step 100/1000 | Loss: 0.8234 | LR: 1.95e-05

Epoch 1 completed in 245.32s
Train - Loss: 0.7123 | Acc: 0.7845 | F1: 0.7654
Val   - Loss: 0.6234 | Acc: 0.8123 | F1: 0.8012 | Precision: 0.8234 | Recall: 0.7845

✓ Saved best model with F1: 0.8012
```

## Evaluation

### Automatic Evaluation

The training script automatically evaluates on the test set after training:

```bash
python scripts/train_baseline.py
```

### Evaluate Existing Model

Evaluate a previously trained model:

```bash
python scripts/train_baseline.py --skip-training
```

### Evaluation Metrics

The model reports:
- **Overall metrics**: Accuracy, Precision, Recall, F1-score
- **Per-class metrics**: Precision, Recall, F1-score for each vulnerability type
- **Confusion matrix**: Shows prediction distribution
- **Classification report**: Detailed breakdown by class

Example output:

```
Test Set Evaluation Results
============================================================

Overall Metrics:
  Accuracy:  0.9234
  Precision: 0.9345
  Recall:    0.9123
  F1-Score:  0.9232

Per-Class Metrics:
  reentrancy:
    Precision: 0.9456
    Recall:    0.9234
    F1-Score:  0.9344
    Support:   150
  ...
```

## Model Outputs

### Saved Files

After training, the following files are saved in `models/codebert/`:

1. **checkpoint_best.pt**: Best model based on validation F1-score
2. **checkpoint_latest.pt**: Latest model checkpoint
3. **model_final.pt**: Final model with test results
4. **training_results.json**: Training metrics and best epoch
5. **test_results.json**: Detailed test set evaluation

### Loading Trained Model

```python
import torch
from src.ml.models import OptimizedCodeBERT

# Load model
checkpoint = torch.load('models/codebert/checkpoint_best.pt')
model = OptimizedCodeBERT(num_labels=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
code = "function withdraw() public { msg.sender.call.value(balance)(); balance = 0; }"

inputs = tokenizer(code, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
predictions = model.predict(inputs['input_ids'], inputs['attention_mask'])

label_names = ['reentrancy', 'timestamp_dependency', 'unchecked_call', 'tx_origin_misuse', 'safe']
print(f"Predicted: {label_names[predictions[0]]}")
```

## Expected Performance

Based on the Lightning Cat paper, the Optimized-CodeBERT model should achieve:

- **F1-Score**: ~93.5%
- **Precision**: ~96.8%
- **Recall**: ~93.6%

These metrics were achieved on the SolidiFI-benchmark dataset with 9,369 injected vulnerabilities.

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `configs/train.yaml`:

```yaml
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 2  # Accumulate gradients
```

### Slow Training

- Use GPU if available (automatically detected)
- Reduce `max_len` to 256 or 384 if functions are typically shorter
- Enable gradient accumulation for larger effective batch size

### Poor Performance

- Ensure data quality and balanced labels
- Use `--use-class-weights` for imbalanced data
- Increase training epochs
- Try different learning rates (1e-5 to 5e-5)

## Advanced Usage

### Custom Model Architecture

Modify the model in `src/ml/models.py`:

```python
# Example: Add more layers to classification head
self.classifier = nn.Sequential(
    nn.Linear(self.hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(256, num_labels)
)
```

### Attention Visualization

Extract attention weights for interpretability:

```python
attention_weights = model.get_attention_weights(input_ids, attention_mask)
# attention_weights: [batch_size, num_heads, seq_len, seq_len]
```

### Fine-tuning Strategy

For limited data, freeze encoder and train only the classification head:

```yaml
model:
  freeze_encoder: true  # Add this to config
```

Then unfreeze after a few epochs for full fine-tuning.

## References

- **Paper**: "Deep learning based solution for smart contract vulnerabilities detection"
- **CodeBERT**: https://github.com/microsoft/CodeBERT
- **Lightning Cat Framework**: Optimized-CodeBERT, Optimized-LSTM, Optimized-CNN

## Support

For issues or questions:
1. Check the configuration in `configs/train.yaml`
2. Verify data format matches expected JSONL structure
3. Review training logs in `logs/` directory
4. Check TensorBoard for training curves
