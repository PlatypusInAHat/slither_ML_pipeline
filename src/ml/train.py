import torch, math, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def bce_loss():
    return nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(model, loader, device, average="macro"):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        logits = model(ids, att)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        y_true.append(y.cpu().numpy()); y_pred.append(preds.cpu().numpy())
    y_true = np.vstack(y_true); y_pred = np.vstack(y_pred)
    # chuyển one-hot → nhãn đơn (mỗi mẫu 1 lớp) cho accuracy
    y_true_arg = y_true.argmax(1); y_pred_arg = y_pred.argmax(1)
    acc = accuracy_score(y_true_arg, y_pred_arg)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def train_loop(model, train_loader, val_loader, epochs, lr, weight_decay, gamma, device, use_amp=True):
    model.to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    loss_fn = bce_loss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(ids, att)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))
        scheduler.step()
        val = evaluate(model, val_loader, device)
        print(f"[VAL] acc={val['accuracy']:.4f} p={val['precision']:.4f} r={val['recall']:.4f} f1={val['f1']:.4f}")
