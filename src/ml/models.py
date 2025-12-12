"""
Optimized-CodeBERT model for vulnerability detection.
Based on the Lightning Cat framework from the research paper.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Optional, Tuple


class OptimizedCodeBERT(nn.Module):
    """
    Optimized-CodeBERT model for smart contract vulnerability detection.
    
    Architecture:
    1. Pre-trained CodeBERT (RoBERTa-based) encoder with multi-head self-attention
    2. Fully connected classification head with dropout
    3. Output layer for 5-class classification
    
    This is the best-performing model from the Lightning Cat paper
    with F1-score of 93.53%.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 5,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 256,
        freeze_encoder: bool = False
    ):
        """
        Initialize the Optimized-CodeBERT model.
        
        Args:
            model_name: Name of the pre-trained CodeBERT model
            num_labels: Number of output classes (5: 4 vulnerabilities + safe)
            dropout: Dropout probability for regularization
            classifier_hidden_dim: Hidden dimension of the classification head
            freeze_encoder: Whether to freeze the encoder weights
        """
        super(OptimizedCodeBERT, self).__init__()
        
        self.num_labels = num_labels
        
        # Load pre-trained CodeBERT (based on RoBERTa)
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for base model
        
        # Optionally freeze encoder weights for faster training
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        # Following the paper: FC layers with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_labels)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classification head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)
            
        Returns:
            Tuple of (logits, loss)
            - logits: [batch_size, num_labels]
            - loss: Scalar tensor (if labels provided)
        """
        # Pass through CodeBERT encoder
        # The encoder uses multi-head self-attention internally
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation (first token)
        # This is the pooled output representing the entire sequence
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Pass through classification head
        logits = self.classifier(cls_output)  # [batch_size, num_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return logits, loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions without computing loss.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Predicted class indices [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        return predictions
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Extract attention weights from a specific layer.
        Useful for interpretability and visualization.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            layer: Which layer to extract attention from (-1 for last layer)
            
        Returns:
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            # outputs.attentions is a tuple of attention weights for each layer
            return outputs.attentions[layer]


class OptimizedCodeBERTWithWeightedLoss(OptimizedCodeBERT):
    """
    Extended version of OptimizedCodeBERT with weighted loss
    for handling class imbalance.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 5,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 256,
        class_weights: Optional[torch.Tensor] = None,
        freeze_encoder: bool = False
    ):
        """
        Initialize the model with weighted loss.
        
        Args:
            model_name: Name of the pre-trained CodeBERT model
            num_labels: Number of output classes
            dropout: Dropout probability
            classifier_hidden_dim: Hidden dimension of classification head
            class_weights: Weights for each class [num_labels]
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            classifier_hidden_dim=classifier_hidden_dim,
            freeze_encoder=freeze_encoder
        )
        
        self.class_weights = class_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with weighted loss.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)
            
        Returns:
            Tuple of (logits, loss)
        """
        # Pass through encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        # Calculate weighted loss if labels are provided
        loss = None
        if labels is not None:
            # Move class weights to the same device as logits
            weights = self.class_weights
            if weights is not None:
                weights = weights.to(logits.device)
            
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits, labels)
        
        return logits, loss


def create_model(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to create a model from configuration.
    
    Args:
        config: Model configuration dictionary
        class_weights: Optional class weights for imbalanced data
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    if class_weights is not None:
        model = OptimizedCodeBERTWithWeightedLoss(
            model_name=model_config.get('name', 'microsoft/codebert-base'),
            num_labels=model_config.get('num_labels', 5),
            dropout=model_config.get('dropout', 0.1),
            classifier_hidden_dim=model_config.get('classifier_hidden_dim', 256),
            class_weights=class_weights,
            freeze_encoder=model_config.get('freeze_encoder', False)
        )
    else:
        model = OptimizedCodeBERT(
            model_name=model_config.get('name', 'microsoft/codebert-base'),
            num_labels=model_config.get('num_labels', 5),
            dropout=model_config.get('dropout', 0.1),
            classifier_hidden_dim=model_config.get('classifier_hidden_dim', 256),
            freeze_encoder=model_config.get('freeze_encoder', False)
        )
    
    return model


if __name__ == "__main__":
    # Test model instantiation
    print("Testing OptimizedCodeBERT model...")
    
    model = OptimizedCodeBERT(num_labels=5)
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 512
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 5, (batch_size,))
    
    logits, loss = model(input_ids, attention_mask, labels)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test prediction
    predictions = model.predict(input_ids, attention_mask)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions: {predictions}")
    
    print("\n✓ Model test passed!")
