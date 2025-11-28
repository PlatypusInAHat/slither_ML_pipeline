import torch
import torch.nn as nn
from transformers import AutoModel

class CodeBERTClassifier(nn.Module):
    def __init__(self, pretrained="microsoft/codebert-base", num_classes=4, dropout=0.1, mlp_hidden=100, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        hidden = self.encoder.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
        # Initialize output layer with small weights for multi-label stability
        with torch.no_grad():
            self.classifier[-1].weight.mul_(0.01)
    
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # dùng [CLS] (pooler) nếu có, else lấy mean-pooling
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            mask = attention_mask.unsqueeze(-1)
            h = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(h)

class LSTMClassifier(nn.Module):
    def __init__(self, pretrained="microsoft/codebert-base", num_classes=4,
                 hidden_size=128, num_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        for p in self.encoder.parameters(): p.requires_grad = False  # freeze
        in_dim = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional,
                            batch_first=True, dropout=dropout)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,L,768]
        h, _ = self.lstm(enc)                       # [B,L,H*dir]
        h_avg = h.mean(dim=1)                       # average pooling theo paper mô tả
        return self.fc(h_avg)

class CNNClassifier(nn.Module):
    def __init__(self, pretrained="microsoft/codebert-base", num_classes=4,
                 channels=(256,128,64), kernel_size=3, pool_kernel=2, fc=(32,), dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        for p in self.encoder.parameters(): p.requires_grad = False  # freeze
        in_dim = self.encoder.config.hidden_size  # 768
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, c1, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(pool_kernel)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.ModuleList()
        last = c3
        for f in fc:
            self.head.append(nn.Linear(last, f)); self.head.append(nn.ReLU()); self.head.append(nn.Dropout(dropout))
            last = f
        self.out = nn.Linear(last, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,L,768]
        x = x.transpose(1, 2)             # [B,768,L] → Conv1d expects CxL
        x = self.conv(x)                   # [B,C',L']
        x = x.max(dim=-1).values           # global max-pool
        x = self.dropout(x)
        for layer in self.head: x = layer(x)
        return self.out(x)
