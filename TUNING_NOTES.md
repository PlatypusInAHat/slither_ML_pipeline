# Model Tuning Summary - Smart Contract Vulnerability Detection

## Mục đích
Tối ưu hóa mô hình phát hiện lỗ hổng trong smart contract (multi-label classification) với 4 loại:
- `reentrancy`
- `timestamp_dependency`
- `unchecked_call`
- `tx_origin_misuse`

---

## Thay đổi chính

### 1. **Sửa Evaluation Metrics** (`src/ml/train.py`)
**Vấn đề**: Đang dùng `argmax()` chuyển đổi multi-label → single-label sai lệch  
**Giải pháp**:
- Loại bỏ `accuracy` (không phù hợp multi-label)
- Giữ Precision/Recall/F1 với multi-hot encoding trực tiếp
- Thêm `hamming_loss` để đánh giá tỷ lệ nhãn sai

### 2. **Điều chỉnh Hyperparameter** (`configs/train.yaml`)
| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `lr` | 1.0e-3 | 2.0e-5 | Chuẩn fine-tuning CodeBERT, tránh overfit |
| `batch_size` | 128 | 32 | Gradient chất lượng cao hơn, ổn định hơn |
| `weight_decay` | 1.0e-4 | 1.0e-2 | L2 regularization mạnh hơn |
| `gamma` | 0.98 | 0.95 | LR decay ổn định |

### 3. **Xử lý Class Imbalance** (`scripts/train_baseline.py`)
- Tính `pos_weight` từ train set: `neg_count / pos_count` cho mỗi lớp
- Truyền vào loss function `BCEWithLogitsLoss(pos_weight=...)`
- Tự động cân bằng các nhãn hiếm gặp

### 4. **Khởi tạo Model** (`src/ml/models.py`)
- Thêm weight scaling nhỏ (×0.01) cho output layer
- Giúp model "cẩn thận" hơn ở giai đoạn đầu training

### 5. **Cấu hình Loss Function** (`src/ml/train.py`)
```python
def bce_loss(pos_weight=None):
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## Cách chạy

```bash
# Chạy với cấu hình mới
python scripts/train_baseline.py --config configs/train.yaml --labels configs/labels.yaml
```

**Kỳ vọng**:
- Precision/Recall/F1 cao hơn trên test set
- Khẩn học ổn định hơn (learning curve smooth)
- Ít overfit hơn do LR thấp + weight_decay cao

---

## Thống kê
- **Files sửa**: 4 files
- **Dòng thay đổi**: ~50 dòng
- **Tính năng mới**: Class weighting, Multi-label metrics
