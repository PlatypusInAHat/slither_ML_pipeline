from torch.utils.data import Dataset
import torch, json
from ..utils.io import jsonl_reader

class HFJsonlDataset(Dataset):
    def __init__(self, jsonl_path, label2id):
        self.samples = []
        self.label2id = label2id
        for ex in jsonl_reader(jsonl_path):
            # one-hot 4 chiều (BCEWithLogitsLoss)
            y = [0]*len(label2id)
            y[label2id[ex["label"]]] = 1
            self.samples.append((
                torch.tensor(ex["input_ids"], dtype=torch.long),
                torch.tensor(ex["attention_mask"], dtype=torch.long),
                torch.tensor(y, dtype=torch.float)
            ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_ids, attn, y = self.samples[idx]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": y}
