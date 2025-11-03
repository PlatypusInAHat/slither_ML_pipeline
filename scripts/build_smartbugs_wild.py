import os
import csv
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

# ==== CẤU HÌNH ====
CSV_PATH = r"D:\datasets\smartbugs-wild.csv"   # <-- sửa theo chỗ bạn để CSV
BASE_DIR = Path(r"D:\slither_hf")
OUT_PATH  = BASE_DIR / "smartbugs_wild.jsonl"
SRC_ROOT  = Path(r"D:\datasets\smartbugs-wild-sources")  # nếu csv chỉ lưu path thì đọc ở đây

BASE_DIR.mkdir(parents=True, exist_ok=True)

# cache HF để đỡ đầy C
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache"

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
MAX_TOKENS = 510

# 4 label mục tiêu
TARGET_LABELS = {
    "reentrancy",
    "timestamp_dependency",
    "unchecked_call",
    "tx_origin_misuse",
}

# ====== PRAGMA FILTER (giữ với 0.8.17) ======
PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)

def _to_ver(tok: str):
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
    TARGET_MAJOR, TARGET_MINOR = 0, 8
    toks = pragma_raw.replace("&&", " ").split()
    ok = False
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue

        # số trần: 0.8.9
        if tok[0].isdigit() or tok.startswith(("^", "=")):
            v = _to_ver(tok)
            if not v:
                return False
            if v[0] == TARGET_MAJOR and v[1] == TARGET_MINOR:
                ok = True
            else:
                return False
            continue

        # >= ...
        if tok.startswith(">="):
            v = _to_ver(tok)
            if not v:
                return False
            if v[0] > TARGET_MAJOR or (v[0] == TARGET_MAJOR and v[1] >= TARGET_MINOR):
                ok = True
            else:
                return False
            continue

        # < ... / <= ...
        if tok.startswith("<") or tok.startswith("<="):
            v = _to_ver(tok)
            if not v:
                return False
            # đa số <0.9.0 thì ok
            if v[0] == 0 and v[1] < 8:
                return False
            if v[0] == 0 and v[1] == 8 and v[2] == 0:
                return False
            continue

        # token lạ
        return False

    return ok


# ====== chuẩn hoá tên lỗi từ CSV ======
def normalize_bug_name(raw: str) -> str:
    if not raw:
        return ""
    low = raw.lower().strip()

    # các biến thể reentrancy
    if "reentr" in low or "etrnanc" in low:
        return "reentrancy"

    # timestamp
    if "timestamp" in low or "time dependence" in low or "predictable" in low:
        return "timestamp_dependency"

    # unchecked / unhandled / low-level send
    if "unchecked" in low or "unhandled" in low or "lowlevel" in low or "low level" in low or "send" in low:
        return "unchecked_call"

    # tx.origin
    if "tx.origin" in low or "tx origin" in low or "use of tx.origin" in low:
        return "tx_origin_misuse"

    return low.replace(" ", "_")


def strip_comments_and_whitespace(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//.*", "", src)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return " ".join(lines)


def read_source_from_row(row: dict) -> str:
    """
    Cố gắng lấy code Solidity từ 1 dòng CSV.
    Ưu tiên:
    1. cột 'source_code'
    2. cột 'source'
    3. cột 'path' / 'file' / 'contract_path' → đọc từ SRC_ROOT
    """
    for key in ("source_code", "source", "code"):
        if key in row and row[key]:
            return row[key]

    for key in ("path", "file", "contract_path", "source_path"):
        if key in row and row[key]:
            p = SRC_ROOT / row[key]
            if p.exists():
                return p.read_text(encoding="utf-8", errors="ignore")

    return ""


# ====== MAIN ======
count_total = 0
count_kept  = 0
count_no_pragma = 0
count_bad_pragma = 0
count_not_in_target = 0

with OUT_PATH.open("w", encoding="utf-8") as fout, open(CSV_PATH, newline="", encoding="utf-8") as fcsv:
    reader = csv.DictReader(fcsv)
    for row in reader:
        count_total += 1

        src = read_source_from_row(row)
        if not src:
            continue

        # pragma
        m = PRAGMA_RE.search(src)
        if not m:
            count_no_pragma += 1
            continue
        pragma_raw = m.group(1).strip()
        if not pragma_compatible_with_0_8_17(pragma_raw):
            count_bad_pragma += 1
            continue

        # label
        raw_label = row.get("vulnerability") or row.get("bug") or row.get("tool_vulnerability") or ""
        label = normalize_bug_name(raw_label)
        if label not in TARGET_LABELS:
            count_not_in_target += 1
            continue

        clean = strip_comments_and_whitespace(src)
        enc = tokenizer(
            clean,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_TOKENS,
            padding="max_length",
        )

        item = {
            "id": row.get("id") or row.get("name") or f"row_{count_total}",
            "label": label,
            "pragma": pragma_raw,
            "clean_code": clean,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            # có thể lưu thêm metadata gốc
            "raw": row,
        }
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        count_kept += 1

print("DONE.")
print("tổng dòng CSV:", count_total)
print("ghi được:", count_kept)
print("bỏ vì không có pragma:", count_no_pragma)
print("bỏ vì pragma không hợp 0.8.17:", count_bad_pragma)
print("bỏ vì label không nằm trong 4 cái:", count_not_in_target)
