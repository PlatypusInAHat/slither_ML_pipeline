"""
Training module for Optimized-CodeBERT model.
Implements training loop, evaluation, and model checkpointing.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import numpy as np


class Trainer:
    """
    Trainer class for Optimized-CodeBERT model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Extract training config
        train_config = config.get('training', {})
        self.num_epochs = train_config.get('num_epochs', 10)
        self.learning_rate = train_config.get('learning_rate', 2e-5)
        self.weight_decay = train_config.get('weight_decay', 0.01)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        
        # Early stopping config
        es_config = config.get('early_stopping', {})
        self.patience = es_config.get('patience', 3)
        self.min_delta = es_config.get('min_delta', 0.001)
        
        # Paths
        paths_config = config.get('paths', {})
        self.output_dir = Path(paths_config.get('output_dir', 'models/codebert'))
        self.log_dir = Path(paths_config.get('log_dir', 'logs'))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Calculate total training steps
        self.total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = train_config.get('warmup_steps', 500)
        
        # Initialize learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.global_step = 0
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Total epochs: {self.num_epochs}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Output directory: {self.output_dir}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, attention_mask, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log every 100 steps
            if self.global_step % 100 == 0 and (step + 1) % self.gradient_accumulation_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | Step {self.global_step}/{self.total_steps} | "
                      f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate the model on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits, loss = self.model(input_ids, attention_mask, labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with F1: {metrics['f1']:.4f}")
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns:
            Best validation metrics
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate(epoch)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            self.writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
            self.writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
            
            # Print metrics
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f}")
            
            # Check if this is the best model
            is_best = val_metrics['f1'] > self.best_f1 + self.min_delta
            
            if is_best:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
                break
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best F1 score: {self.best_f1:.4f} at epoch {self.best_epoch}")
        print("="*60 + "\n")
        
        self.writer.close()
        
        return {'best_f1': self.best_f1, 'best_epoch': self.best_epoch}


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    label_names: Optional[list] = None
) -> Dict:
    """
    Evaluate model on test set and generate detailed metrics.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        label_names: Names of the labels for classification report
        
    Returns:
        Dictionary containing metrics and predictions
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Classification report
    if label_names is None:
        label_names = ['reentrancy', 'timestamp_dependency', 'unchecked_call', 'tx_origin_misuse', 'safe']
    
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*60)
    print("Test Set Evaluation Results")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, label in enumerate(label_names):
        print(f"  {label}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
        print(f"    Support:   {support[i]}")
    
    print(f"\nClassification Report:")
    print(report)
    
    print(f"\nConfusion Matrix:")
    print(cm)
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'logits': all_logits
    }
