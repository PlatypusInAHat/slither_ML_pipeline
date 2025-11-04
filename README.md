# slither-ml-pipeline

Pipeline tạo **dataset mức hàm Solidity** từ bộ `slither-audited-smart-contracts` (Hugging Face) bằng cách:
1) lọc theo `pragma` (tương thích `solc 0.8.17` mặc định),  
2) chạy **Slither** để phát hiện lỗ hổng,  
3) trích **mã nguồn từng hàm** bị gắn nhãn, làm sạch,  
4) **tokenize** (CodeBERT) → ghi **JSONL** để train ML (multi-label).

> Nhóm nhãn sử dụng (4 loại): `reentrancy`, `timestamp_dependency`, `unchecked_call`, `tx_origin_misuse`.

---

## Mục lục

- [Kiến trúc & Thư mục](#kiến-trúc--thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình đường dẫn](#cấu-hình-đường-dẫn)
- [Chạy pipeline HF](#chạy-pipeline-hf)
- [Resume / chạy tiếp](#resume--chạy-tiếp)
- [Đầu ra](#đầu-ra)
- [Thêm mẫu “safe” (không lỗi)](#thêm-mẫu-safe-không-lỗi)
- [Gợi ý huấn luyện nhanh](#gợi-ý-huấn-luyện-nhanh)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Kiến trúc & Thư mục

```
slither-ml-pipeline/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ environment.yml                # (optional) Conda
├─ pyproject.toml                 # (optional) black/ruff/isort
├─ Makefile                       # (optional) make run, make eda, ...
├─ .env.example                   # ví dụ biến HF_*, cache (đừng commit .env thật)
│
├─ configs/
│  ├─ paths.windows.yaml          # BASE_DIR: D:/slither-ml-pipeline
│  ├─ paths.linux.yaml            # BASE_DIR: /data/slither-ml-pipeline
│  ├─ labels.yaml                 # map/alias nhãn → canonical labels
│  └─ train.yaml                  # siêu tham số huấn luyện
│
├─ docker/
│  ├─ Dockerfile.slither          # (optional) image có nhiều solc
│  └─ compose.yaml                # (optional)
│
├─ data/
│  ├─ raw/
│  │  └─ hf_cache/                # cache HF (khuyến nghị .gitignore)
│  ├─ interim/
│  │  ├─ contracts/               # contract_{idx}.sol đã lọc pragma
│  │  └─ reports/                 # contract_{idx}.json từ Slither
│  └─ processed/
│     ├─ dataset_from_hf.jsonl    # đầu ra để train
│     └─ splits/                  # (optional) train/val/test
│
├─ notebooks/
│  ├─ 01_eda_hf.ipynb
│  ├─ 02_eda_smartbugs.ipynb
│  ├─ 03_build_hf_jsonl.ipynb
│  ├─ 10_train_baseline.ipynb
│  └─ 11_eval_metrics.ipynb
│
├─ scripts/
│  ├─ run_hf_pipeline.py          # entry: load HF → slither → jsonl
│  ├─ convert_jsonl_to_csv.py
│  ├─ parse_smartbugs_csv.py
│  ├─ split_dataset.py
│  └─ train_baseline.py
│
└─ src/
   ├─ utils/
   │  ├─ io.py                    # jsonl_append, read/write tiện ích
   │  ├─ paths.py                 # đọc configs/paths.*.yaml
   │  ├─ labels.py                # TARGET_LABELS, normalize_label()
   │  └─ pragma.py                # pragma_compatible_with_0_8_17()
   ├─ slither_tools/
   │  ├─ runner.py                # gọi slither an toàn
   │  └─ report_parser.py         # parse JSON Slither
   ├─ builders/
   │  └─ hf_pipeline.py           # build dataset từ HF
   ├─ features/
   │  ├─ tokenizer.py             # wrapper CodeBERT (nếu cần)
   │  └─ clean.py                 # strip comments, extract_function_source()
   └─ ml/
      ├─ dataset.py
      ├─ models.py
      └─ train.py
```

---

## Yêu cầu hệ thống

- **Python** ≥ 3.10  
- **Slither** + **solc** (compiler Solidity). Mặc định dùng solc **0.8.17**.  
- **pip/conda** để cài thư viện.  
- Không bắt buộc cài PyTorch/TF nếu chỉ cần tokenizer (nhưng model training sẽ cần).

---

## Cài đặt

### Pip (nhanh)

```bash
pip install -r requirements.txt
```

### Conda (tuỳ chọn)

```bash
conda env create -f environment.yml
conda activate slither-ml
```

> Nếu cần nhiều phiên bản `solc`: khuyến nghị dùng WSL + `solc-select`.  
> Windows native: có thể tải `solc.exe` từng bản và đặt biến `CRYTIC_SOLC` trỏ tới exe phù hợp.

---

## Cấu hình đường dẫn

Sửa `configs/paths.*.yaml` cho phù hợp máy bạn.

**Ví dụ (Windows):**
```yaml
# configs/paths.windows.yaml
BASE_DIR: D:/slither-ml-pipeline
HF_CACHE: D:/hf_cache
```

---

## Chạy pipeline HF

Tạo dataset từ Hugging Face (config `big-multilabel`, split `train`):

```bash
python scripts/run_hf_pipeline.py   --paths configs/paths.windows.yaml   --dataset mwritescode/slither-audited-smart-contracts   --config big-multilabel   --split train   --start_idx 0   --end_idx 200000
```

> Mặc định pipeline sẽ:
> - Lọc theo `pragma` tương thích `0.8.17`
> - Gọi Slither để sinh report (cache theo file)
> - Trích các **hàm** có cảnh báo thuộc 4 nhãn mục tiêu
> - Tokenize (CodeBERT) → ghi vào `data/processed/dataset_from_hf.jsonl`

---

## Resume / chạy tiếp

Có 2 cách:

1) **Theo index**: chỉ định `--start_idx` (ví dụ tiếp tục từ 36,054):
```bash
python scripts/run_hf_pipeline.py --start_idx 36054 --end_idx 200000
```

2) **Chống trùng (orig_id, function)**:  
   Bật tính năng đọc `dataset_from_hf.jsonl` để skip cặp đã tồn tại (có trong code mẫu mở rộng).

> Pipeline mở file **append**, không xoá dữ liệu cũ.

---

## Đầu ra

**`data/processed/dataset_from_hf.jsonl`**, mỗi dòng là 1 hàm:

```json
{
  "orig_id": 960,
  "file": "D:\\...\\contracts\\contract_960.sol",
  "function": "preSaleIsActive",
  "label": "timestamp_dependency",
  "clean_code": "function preSaleIsActive() public view returns (bool) { ... }",
  "input_ids": [...],
  "attention_mask": [...]
}
```

- `orig_id`: chỉ số dòng trong HF dataset (split/config đang dùng).  
- `label`: 1 trong 4 nhãn mục tiêu.  
- `clean_code`: mã đã bỏ comment + gộp 1 dòng.  
- `input_ids`, `attention_mask`: mã hoá từ `microsoft/codebert-base` (độ dài 510).

---

## Thêm mẫu “safe” (không lỗi)

Nếu mục tiêu là **phát hiện có/không lỗi**, cần thêm mẫu âm (safe):
- **Âm từ Slither**: hợp đồng/hàm **không** có 4 cảnh báo ⇒ gán vector `[0,0,0,0]`.
- **Âm trong cùng contract**: các hàm không nằm trong `vuln_map`.
- **Âm nguồn khác**: ví dụ smartbugs-wild (benign), verified contracts.
- **Cân bằng**: 1:1 đến 1:3 (dương:âm) theo batch; cân nhắc **focal loss** / class weights.

> Nếu chỉ **phân loại loại lỗi** (khi đã có lỗi), có thể **không cần** lớp “safe”.

---

## Gợi ý huấn luyện nhanh

- Định dạng nhãn cho **multi-label**: vector 4 chiều (sigmoid + BCEWithLogitsLoss).  
- Baseline:
  - Fine-tune CodeBERT (yêu cầu PyTorch/TF), hoặc
  - Rút **embedding** và train classifier nhẹ (LogReg/SVM).  
- Metric: mAP, macro/micro F1, per-class PR.

---

## Troubleshooting

- **Cảnh báo** “one of PyTorch/TF/Flax not found”:  
  Bỏ qua nếu chỉ dùng tokenizer. Cần train thì cài `torch`/`tensorflow`.

- **solc version mismatch** (ví dụ `pragma solidity 0.8.0` nhưng máy dùng `solc 0.8.17`):  
  - Cách nhanh: **skip** bằng `pragma_compatible_with_0_8_17()` (đã tích hợp).  
  - Cách kỹ: cài đa phiên bản `solc` và chọn theo pragma (WSL + `solc-select`).

- **Invalid SPDX license identifier**:  
  Slither/solc có thể từ chối biên dịch. Pipeline đã `continue` nếu không có report.

- **Nhiều log Slither**:  
  Đã tắt stdout/stderr trong `runner.py`. Tuỳ chỉnh nếu cần debug.

---

## License

Mã nguồn pipeline theo giấy phép của repo này (MIT/Apache-2.0/GPL — tuỳ bạn).  
Dữ liệu gốc thuộc về các bộ **Hugging Face / SmartBugs / …** theo giấy phép của từng nguồn.
