import argparse, yaml, torch
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from src.utils.labels import load_labels
from src.ml.dataset import HFJsonlDataset
from src.ml.models import CodeBERTClassifier, LSTMClassifier, CNNClassifier
from src.ml.train import train_loop, evaluate

def build_model(cfg, num_classes):
    name = cfg["model"]["name"]
    pt = cfg["model"]["pretrained"]
    if name == "codebert":
        return CodeBERTClassifier(pretrained=pt, num_classes=num_classes,
                                  dropout=cfg["model"]["dropout"],
                                  mlp_hidden=cfg["model"]["mlp_hidden"],
                                  freeze_encoder=False)
    elif name == "lstm":
        p = cfg["model"]["lstm"]
        return LSTMClassifier(pretrained=pt, num_classes=num_classes,
                              hidden_size=p["hidden_size"], num_layers=p["num_layers"],
                              bidirectional=p["bidirectional"], dropout=p["dropout"])
    elif name == "cnn":
        p = cfg["model"]["cnn"]
        return CNNClassifier(pretrained=pt, num_classes=num_classes,
                             channels=tuple(p["channels"]),
                             kernel_size=p["kernel_size"],
                             pool_kernel=p["pool_kernel"],
                             fc=tuple(p["fc"]))
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--labels", default="configs/labels.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    label2id, id2label = load_labels(args.labels)
    ds = HFJsonlDataset(cfg["data"]["jsonl_path"], label2id)
    idx = list(range(len(ds)))
    train_idx, tmp_idx = train_test_split(idx, test_size=cfg["data"]["val_ratio"]+cfg["data"]["test_ratio"], random_state=cfg["data"]["seed"])
    val_rel = cfg["data"]["val_ratio"] / (cfg["data"]["val_ratio"]+cfg["data"]["test_ratio"]) if (cfg["data"]["test_ratio"]>0) else 1.0
    val_idx, test_idx = train_test_split(tmp_idx, test_size=1-val_rel, random_state=cfg["data"]["seed"])

    # Tính pos_weight để xử lý class imbalance
    train_subset = Subset(ds, train_idx)
    all_labels = []
    for i in range(len(train_subset)):
        all_labels.append(train_subset[i]["labels"])
    all_labels = np.array(all_labels)
    pos_count = all_labels.sum(axis=0)
    neg_count = len(all_labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-8)  # tránh chia 0

    train_loader = DataLoader(train_subset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])
    test_loader  = DataLoader(Subset(ds, test_idx),  batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    device = cfg["train"]["device"] if torch.cuda.is_available() and cfg["train"]["device"]=="cuda" else "cpu"
    model = build_model(cfg, num_classes=len(label2id))
    use_amp = bool(cfg["train"]["mixed_precision"]) and device=="cuda"

    print(f"[INFO] Class weights: {pos_weight}")
    train_loop(model, train_loader, val_loader,
               epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
               weight_decay=cfg["train"]["weight_decay"], gamma=cfg["train"]["gamma"],
               device=device, use_amp=use_amp, pos_weight=pos_weight)

    print("=== TEST ===")
    metrics = evaluate(model, test_loader, device, average=cfg["metrics"]["average"])
    print(metrics)

if __name__ == "__main__":
    main()
