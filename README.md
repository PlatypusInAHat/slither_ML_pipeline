# slither-ml-pipeline

Pipeline: HF Slither dataset → lọc pragma → chạy Slither → JSONL/CSV → ML.

## Lightning Cat - Optimized-CodeBERT

Triển khai mô hình **Optimized-CodeBERT** từ framework Lightning Cat để phát hiện lỗ hổng bảo mật trong smart contract Solidity.

### Tính năng

- ✅ Phát hiện 4 loại lỗ hổng phổ biến:
  - **Reentrancy**: Tấn công tái nhập
  - **Timestamp Dependency**: Phụ thuộc dấu thời gian
  - **Unchecked Call**: Lỗi không xử lý ngoại lệ
  - **tx.origin Misuse**: Sử dụng tx.origin sai cách
- ✅ Phân loại code **safe** (không có lỗ hổng)
- ✅ Sử dụng CodeBERT pre-trained model
- ✅ Kiến trúc Transformer với multi-head self-attention
- ✅ Hỗ trợ class weights cho dữ liệu mất cân bằng
- ✅ Early stopping và model checkpointing
- ✅ TensorBoard logging

### Cài đặt

```bash
pip install -r requirements.txt
```

### Sử dụng nhanh

1. **Chuẩn bị dữ liệu (Pipeline hợp nhất):**
```bash
# Trích xuất từ HuggingFace dataset, lọc pragma, chạy Slither, 
# và tạo dataset với cả vuln + safe trong một bước
python scripts/run_hf_pipeline.py
```

2. **Huấn luyện mô hình:**
```bash
python scripts/train_baseline.py
```

3. **Theo dõi training:**
```bash
tensorboard --logdir logs
```

### Cấu trúc dự án

```
slither-ml-pipeline/
├── configs/
│   ├── labels.yaml          # Cấu hình nhãn (4 lỗ hổng + safe)
│   └── train.yaml           # Cấu hình training
├── src/
│   └── ml/
│       ├── dataset.py       # Dataset loader với CodeBERT tokenizer
│       ├── models.py        # Optimized-CodeBERT model
│       └── train.py         # Training infrastructure
├── scripts/
│   ├── run_hf_pipeline.py   # Pipeline hợp nhất: HF → Slither → Vuln + Safe
│   └── train_baseline.py   # Script huấn luyện chính
└── USAGE.md                 # Hướng dẫn chi tiết
```

### Hiệu suất mong đợi

Dựa trên nghiên cứu Lightning Cat:
- **F1-Score**: ~93.5%
- **Precision**: ~96.8%
- **Recall**: ~93.6%

### Tài liệu

Xem [USAGE.md](USAGE.md) để biết hướng dẫn chi tiết về:
- Chuẩn bị dữ liệu
- Cấu hình training
- Đánh giá mô hình
- Sử dụng mô hình đã train
- Troubleshooting

### Tham khảo

- **Paper**: "Deep learning based solution for smart contract vulnerabilities detection"
- **CodeBERT**: https://github.com/microsoft/CodeBERT
- **Framework**: Lightning Cat (Optimized-CodeBERT)
