import os
import json
import subprocess
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


REPO_ROOT = Path(r"D:\slither-ml-pipeline")
HF_CACHE_ROOT = REPO_ROOT / "cache" / "hf"

# --- cache ---
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)                      # gốc cho HF Hub
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")         # cache file model/dataset từ Hub
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_ROOT / "datasets")  # cache arrow/dataset

dataset = load_dataset(
    "mwritescode/slither-audited-smart-contracts",
    "big-multilabel",
    split="train",
    trust_remote_code=True,
    verification_mode="no_checks",
    cache_dir=r"D:\hf_cache",
)

BASE_DIR = Path(r"D:\slither_hf")
CONTRACT_DIR = BASE_DIR / "contracts"
REPORT_DIR  = BASE_DIR / "reports"
OUT_PATH    = BASE_DIR / "dataset_from_hf.jsonl"

CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
MAX_TOKENS = 510

# chỉ quan tâm 4 loại
TARGET_LABELS = {
    "reentrancy",
    "timestamp_dependency",
    "unchecked_call",     # unhandled-exceptions
    "tx_origin_misuse",
}

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
            # đa số là <0.9.0 → cho qua
            if v[0] == 0 and v[1] < 8:
                return False
        else:
            return False
    return ok

def strip_comments_and_whitespace(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//.*", "", src)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return " ".join(lines)

def normalize_label(name: str) -> str:
    lower = name.lower()
    # 1. reentrancy
    if "reentr" in lower:
        return "reentrancy"
    # 2. timestamp
    if "timestamp" in lower or "time dependence" in lower or "predictable" in lower:
        return "timestamp_dependency"
    # 3. unchecked / unhandled low level
    if "unchecked" in lower or "unhandled" in lower or "low level" in lower or "send" in lower:
        return "unchecked_call"
    # 4. tx.origin
    if "tx.origin" in lower or "tx origin" in lower:
        return "tx_origin_misuse"
    return lower.replace(" ", "_")

def parse_slither_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {}
    detectors = data.get("results", {}).get("detectors", [])
    impact_rank = {"High": 3, "Medium": 2, "Low": 1, "Informational": 0}
    per_file = {}
    for det in detectors:
        label = normalize_label(det.get("check", ""))
        # chỉ quan tâm 4 loại
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
    for fname in list(per_file.keys()):
        per_file[fname] = {fn: lr[0] for fn, lr in per_file[fname].items()}
    return per_file

def extract_function_source(code: str, func_name: str) -> str:
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

count_samples = 0
count_skipped_pragma = 0

with OUT_PATH.open("w", encoding="utf-8") as fout:
    for idx, example in enumerate(dataset):
        if idx >= 1000000:
            break

        src = example["source_code"]

        m = PRAGMA_RE.search(src)
        if not m:
            count_skipped_pragma += 1
            continue
        if not pragma_compatible_with_0_8_17(m.group(1).strip()):
            count_skipped_pragma += 1
            continue

        sol_path = CONTRACT_DIR / f"contract_{idx}.sol"
        sol_path.write_text(src, encoding="utf-8")

        report_path = REPORT_DIR / f"contract_{idx}.json"
        cmd = [
            "python", "-m", "slither",
            str(sol_path),
            "--json", str(report_path),
        ]
        subprocess.run(cmd, check=False)

        if not report_path.exists():
            continue

        vuln_map_all = parse_slither_json(report_path)
        if not vuln_map_all:
            continue

        # lấy file trùng tên trước
        keys = [str(sol_path), sol_path.name, f"contracts/contract_{idx}.sol"]
        vuln_map = None
        for k in keys:
            if k in vuln_map_all:
                vuln_map = vuln_map_all[k]
                break
        if vuln_map is None:
            first_file = next(iter(vuln_map_all.keys()))
            vuln_map = vuln_map_all[first_file]

        # đến đây chỉ còn 4 label
        for func_name, label in vuln_map.items():
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
            fout.write(json.dumps({
                "orig_id": idx,
                "file": str(sol_path),
                "function": func_name,
                "label": label,
                "clean_code": clean,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }, ensure_ascii=False) + "\n")
            count_samples += 1

print("DONE.")
print("samples ghi được:", count_samples)
print("bỏ vì pragma:", count_skipped_pragma)
