"""
Pipeline hợp nhất: HF Dataset → Lọc pragma → Chạy Slither → Trích xuất vuln + safe → JSONL

Script này kết hợp:
1. Trích xuất contracts từ HuggingFace dataset
2. Lọc theo pragma solidity 0.8.x
3. Chạy Slither để phát hiện lỗ hổng
4. Trích xuất functions có lỗ hổng (4 loại: reentrancy, timestamp_dependency, unchecked_call, tx_origin_misuse)
5. Trích xuất functions SAFE (không có bất kỳ issue nào theo Slither)
6. Tokenize bằng CodeBERT và lưu vào JSONL
"""

import os
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Set, List

from datasets import load_dataset
from transformers import AutoTokenizer


# =========================
# CẤU HÌNH CƠ BẢN
# =========================

# Chỉ duyệt các contract có idx trong [START_IDX, END_IDX)
# Để xử lý toàn bộ dataset, đặt START_IDX = 0
# Script sẽ tự động phát hiện contract cuối cùng đã xử lý và tiếp tục
START_IDX = 0
END_IDX = 200000

# Giới hạn token cho CodeBERT
MAX_TOKENS = 512

# Gốc repo
REPO_ROOT = Path(r"D:\slither-ml-pipeline").resolve()

# Thư mục cache cho HuggingFace
HF_CACHE_ROOT = REPO_ROOT / "cache" / "hf"

# Biến môi trường cache cho HF
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_ROOT / "datasets")

# Thư mục intermediate cho contract & report Slither
DATA_INTERIM = REPO_ROOT / "data" / "interim" / "slither"
CONTRACT_DIR = DATA_INTERIM / "contracts"
REPORT_DIR = DATA_INTERIM / "reports"

# Thư mục output cho dataset đã xử lý
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUT_PATH = DATA_PROCESSED / "dataset_with_safe.jsonl"

# Tạo thư mục nếu chưa có
CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# 4 loại lỗ hổng mục tiêu + safe
TARGET_LABELS = {
    "reentrancy",
    "timestamp_dependency",
    "unchecked_call",
    "tx_origin_misuse",
}
SAFE_LABEL = "safe"

# Regex
PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
FUNC_DEF_RE = re.compile(r"\bfunction\s+([A-Za-z_]\w*)\s*\(")

# Khởi tạo tokenizer CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


# =========================
# HÀM XỬ LÝ PRAGMA
# =========================

def _to_ver(tok: str):
    """Chuyển đổi token version thành tuple (major, minor, patch)."""
    tok = tok.strip()
    for pre in ("^", "=", ">=", "<=", ">", "<"):
        if tok.startswith(pre):
            tok = tok[len(pre):].strip()
            break
    if not tok or not tok[0].isdigit():
        return None
    parts = tok.split(".")
    parts += ["0"] * (3 - len(parts))
    try:
        return tuple(int(p) for p in parts[:3])
    except ValueError:
        return None


def pragma_compatible_with_0_8_17(pragma_raw: str) -> bool:
    """Kiểm tra pragma có tương thích với 0.8.x không."""
    TARGET_MAJOR, TARGET_MINOR = 0, 8
    toks = pragma_raw.replace("&&", " ").split()
    ok = False
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        if tok[0].isdigit() or tok.startswith(("^", "=")):
            v = _to_ver(tok)
            if not v:
                return False
            if v[0] == TARGET_MAJOR and v[1] == TARGET_MINOR:
                ok = True
            else:
                return False
        elif tok.startswith(">="):
            v = _to_ver(tok)
            if not v:
                return False
            if v[0] > TARGET_MAJOR or (v[0] == TARGET_MAJOR and v[1] >= TARGET_MINOR):
                ok = True
            else:
                return False
        elif tok.startswith("<") or tok.startswith("<="):
            v = _to_ver(tok)
            if not v:
                return False
            if v[0] == 0 and v[1] < 8:
                return False
        else:
            return False
    return ok


# =========================
# CLEAN CODE
# =========================

def strip_comments_and_whitespace(src: str) -> str:
    """Loại bỏ comment /* */ và //, gom code về 1 dòng."""
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//.*", "", src)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return " ".join(lines)


def normalize_label(name: str) -> str:
    """Chuẩn hóa tên detector Slither về 4 label chính."""
    lower = name.lower()
    if "reentr" in lower:
        return "reentrancy"
    if "timestamp" in lower or "time dependence" in lower or "predictable" in lower:
        return "timestamp_dependency"
    if "unchecked" in lower or "unhandled" in lower or "low level" in lower or "send" in lower:
        return "unchecked_call"
    if "tx.origin" in lower or "tx origin" in lower:
        return "tx_origin_misuse"
    return lower.replace(" ", "_")


# =========================
# PARSE SLITHER JSON
# =========================

def parse_slither_vulnerabilities(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Đọc report Slither JSON và trả về vulnerabilities:
    {
        "filename.sol": {
            "funcName": "reentrancy",
            ...
        },
        ...
    }
    Chỉ giữ lại label trong TARGET_LABELS, ưu tiên impact cao hơn.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {}

    detectors = data.get("results", {}).get("detectors", [])
    impact_rank = {"High": 3, "Medium": 2, "Low": 1, "Informational": 0}
    per_file: Dict[str, Dict[str, Any]] = {}

    for det in detectors:
        label = normalize_label(det.get("check", ""))
        if label not in TARGET_LABELS:
            continue
        impact = det.get("impact", "Low")
        for elem in det.get("elements", []):
            func = elem.get("name")
            sm = elem.get("source_mapping", {})
            filename = sm.get("filename_relative") or sm.get("filename_absolute")
            if not func or not filename:
                continue
            cur = per_file.setdefault(filename, {})
            rank = impact_rank.get(impact, 0)
            old = cur.get(func)
            if old is None or rank > old[1]:
                cur[func] = (label, rank)

    # Chỉ giữ lại label, bỏ rank
    for fname in list(per_file.keys()):
        per_file[fname] = {fn: lr[0] for fn, lr in per_file[fname].items()}

    return per_file


def parse_slither_all_issues(path: Path) -> Dict[str, Set[str]]:
    """
    Đọc report Slither JSON và trả về TẤT CẢ functions có bất kỳ issue nào:
    {
        "filename.sol": {"func1", "func2", ...},
        ...
    }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {}

    detectors = data.get("results", {}).get("detectors", [])
    per_file: Dict[str, Set[str]] = {}

    for det in detectors:
        for elem in det.get("elements", []):
            func = elem.get("name")
            sm = elem.get("source_mapping", {}) or {}
            filename = sm.get("filename_relative") or sm.get("filename_absolute")
            if not func or not filename:
                continue
            per_file.setdefault(filename, set()).add(func)

    return per_file


# =========================
# EXTRACT FUNCTION SOURCE
# =========================

def extract_function_source(code: str, func_name: str) -> str:
    """
    Tìm đoạn source của function func_name trong code Solidity,
    bao gồm toàn bộ thân hàm từ 'function ... {' đến '}' cuối cùng.
    """
    pat = re.compile(
        r"\bfunction\s+" + re.escape(func_name) + r"\s*\([^)]*\)\s*[^{;]*\{",
        re.DOTALL,
    )
    m = pat.search(code)
    if not m:
        return ""
    start_brace = code.find("{", m.start())
    depth = 0
    end_idx = None
    for i in range(start_brace, len(code)):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        return ""
    return code[m.start(): end_idx + 1]


def list_all_functions(code: str) -> List[str]:
    """Liệt kê tất cả function name trong source."""
    return sorted({m.group(1) for m in FUNC_DEF_RE.finditer(code)})


def get_last_processed_idx() -> int:
    """
    Đọc file output JSONL và tìm orig_id lớn nhất đã xử lý.
    Trả về -1 nếu file chưa tồn tại hoặc rỗng.
    """
    if not OUT_PATH.exists():
        return -1
    
    max_idx = -1
    try:
        with OUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    orig_id = obj.get("orig_id", -1)
                    if orig_id > max_idx:
                        max_idx = orig_id
    except Exception as e:
        print(f"[WARN] Không thể đọc file output: {e}")
        return -1
    
    return max_idx



# =========================
# MAIN PIPELINE
# =========================

def main():
    print("="*80)
    print("PIPELINE HỢP NHẤT: HF Dataset → Lọc pragma → Slither → Vuln + Safe → JSONL")
    print("="*80)
    print()
    
    # Tìm contract cuối cùng đã xử lý
    last_processed = get_last_processed_idx()
    
    # Xác định điểm bắt đầu thực tế
    actual_start = max(START_IDX, last_processed + 1)
    
    if last_processed >= 0:
        print(f"✓ Phát hiện file output đã tồn tại: {OUT_PATH}")
        print(f"✓ Contract cuối cùng đã xử lý: {last_processed}")
        print(f"✓ Sẽ tiếp tục từ contract: {actual_start}")
        print()
    else:
        print(f"✓ Bắt đầu xử lý mới từ contract: {actual_start}")
        print()
    
    # Load dataset từ HuggingFace
    print("Đang tải dataset từ HuggingFace...")
    dataset = load_dataset(
        "mwritescode/slither-audited-smart-contracts",
        "big-multilabel",
        split="train",
        trust_remote_code=True,
        verification_mode="no_checks",
        cache_dir=str(HF_CACHE_ROOT),
    )
    print(f"Dataset loaded: {len(dataset)} contracts")
    print()

    count_vuln_samples = 0
    count_safe_samples = 0
    count_skipped_pragma = 0
    count_slither_failed = 0
    
    # Theo dõi các (file, function) đã ghi để tránh duplicate
    written_pairs: Set[tuple] = set()
    
    # Load các pairs đã ghi từ file cũ (nếu có)
    if last_processed >= 0:
        print("Đang load các function đã xử lý để tránh duplicate...")
        try:
            with OUT_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        written_pairs.add((obj["file"], obj["func_name"]))
            print(f"✓ Đã load {len(written_pairs)} function pairs")
            print()
        except Exception as e:
            print(f"[WARN] Không thể load written_pairs: {e}")
            print()

    # MỞ FILE Ở CHẾ ĐỘ APPEND thay vì ghi đè
    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            if idx < actual_start:
                continue
            if idx >= END_IDX:
                break

            if idx % 100 == 0:
                print(f"Processing contract {idx}/{END_IDX} (started from {actual_start})...")


            src = example["source_code"]

            # Lọc theo pragma solidity
            m = PRAGMA_RE.search(src)
            if not m:
                count_skipped_pragma += 1
                continue
            if not pragma_compatible_with_0_8_17(m.group(1).strip()):
                count_skipped_pragma += 1
                continue

            # Ghi contract .sol
            sol_path = CONTRACT_DIR / f"contract_{idx}.sol"
            if not sol_path.exists():
                sol_path.write_text(src, encoding="utf-8")

            # Đường dẫn report Slither
            report_path = REPORT_DIR / f"contract_{idx}.json"

            # Chỉ chạy Slither nếu chưa có report
            if not report_path.exists():
                cmd = [
                    "python",
                    "-m",
                    "slither",
                    str(sol_path),
                    "--json",
                    str(report_path),
                ]
                try:
                    subprocess.run(cmd, check=False, timeout=300)
                except subprocess.TimeoutExpired:
                    print(f"[WARN] Slither timeout cho contract_{idx}.sol")
                    count_slither_failed += 1
                    continue

            if not report_path.exists():
                count_slither_failed += 1
                continue

            # Parse vulnerabilities (4 loại TARGET_LABELS)
            vuln_map_all = parse_slither_vulnerabilities(report_path)
            
            # Parse tất cả issues (để xác định safe functions)
            all_issues_map = parse_slither_all_issues(report_path)

            # Map đúng key file trong report
            keys = [
                str(sol_path),
                sol_path.name,
                f"contracts/contract_{idx}.sol",
            ]
            
            vuln_map = None
            all_issues = set()
            
            for k in keys:
                if k in vuln_map_all:
                    vuln_map = vuln_map_all[k]
                if k in all_issues_map:
                    all_issues = all_issues_map[k]
                if vuln_map is not None and all_issues:
                    break
            
            # Nếu không tìm thấy key, lấy file đầu tiên
            if vuln_map is None and vuln_map_all:
                first_file = next(iter(vuln_map_all.keys()))
                vuln_map = vuln_map_all[first_file]
            
            if not all_issues and all_issues_map:
                first_file = next(iter(all_issues_map.keys()))
                all_issues = all_issues_map[first_file]

            # ===== GHI CÁC FUNCTION CÓ LỖ HỔNG =====
            if vuln_map:
                for func_name, label in vuln_map.items():
                    if label not in TARGET_LABELS:
                        continue
                    
                    pair = (str(sol_path), func_name)
                    if pair in written_pairs:
                        continue
                    
                    raw_func = extract_function_source(src, func_name)
                    if not raw_func:
                        continue
                    
                    clean = strip_comments_and_whitespace(raw_func)
                    enc = tokenizer(
                        clean,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=MAX_TOKENS,
                        padding="max_length",
                    )
                    
                    obj = {
                        "orig_id": idx,
                        "file": str(sol_path),
                        "func_name": func_name,
                        "label": label,
                        "code": clean,
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                    }
                    
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written_pairs.add(pair)
                    count_vuln_samples += 1

            # ===== GHI CÁC FUNCTION SAFE =====
            all_funcs = list_all_functions(src)
            
            for func_name in all_funcs:
                pair = (str(sol_path), func_name)
                
                # Bỏ qua nếu đã ghi (là vuln)
                if pair in written_pairs:
                    continue
                
                # Bỏ qua nếu có bất kỳ issue nào
                if func_name in all_issues:
                    continue
                
                # Function này là SAFE
                raw_func = extract_function_source(src, func_name)
                if not raw_func:
                    continue
                
                clean = strip_comments_and_whitespace(raw_func)
                enc = tokenizer(
                    clean,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    padding="max_length",
                )
                
                obj = {
                    "orig_id": idx,
                    "file": str(sol_path),
                    "func_name": func_name,
                    "label": SAFE_LABEL,
                    "code": clean,
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }
                
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written_pairs.add(pair)
                count_safe_samples += 1

    print()
    print("="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    print(f"Tổng số samples có lỗ hổng: {count_vuln_samples}")
    print(f"Tổng số samples safe: {count_safe_samples}")
    print(f"Tổng số samples: {count_vuln_samples + count_safe_samples}")
    print(f"Bỏ qua do pragma không tương thích: {count_skipped_pragma}")
    print(f"Slither failed: {count_slither_failed}")
    print(f"Output file: {OUT_PATH}")
    print()
    
    # Thống kê phân bố labels
    print("Đang tính phân bố labels...")
    label_counts = {}
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                label = obj.get("label", "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nPhân bố labels:")
    total = sum(label_counts.values())
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:25s}: {count:6d} ({pct:5.2f}%)")
    print(f"  {'TOTAL':25s}: {total:6d}")
    print()


if __name__ == "__main__":
    main()
