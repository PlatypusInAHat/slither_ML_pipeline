# src/builders/hf_pipeline.py
import os, json, subprocess, re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils.paths import PathsConfig
from src.utils.io import jsonl_append
from src.utils.labels import TARGET_LABELS, normalize_label
from src.features.clean import strip_comments_and_whitespace, extract_function_source
from src.utils.pragma import pragma_compatible_with_0_8_17
from src.slither_tools.runner import run_slither_once
from src.slither_tools.report_parser import parse_slither_json

PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
MAX_TOKENS = 510

def build_from_hf(
    paths_yaml: str,
    labels_yaml: str,
    dataset_id: str,
    dataset_config: str,
    split: str,
    start_idx: int,
    end_idx: int,
    hf_cache_override: str | None = None,
):
    # === paths & env ===
    paths = PathsConfig.from_yaml(paths_yaml)
    BASE_DIR = Path(paths.base_dir)
    CONTRACT_DIR = BASE_DIR / "data" / "interim" / "contracts"
    REPORT_DIR   = BASE_DIR / "data" / "interim" / "reports"
    OUT_PATH     = BASE_DIR / "data" / "processed" / "dataset_from_hf.jsonl"

    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # HF cache (ưu tiên arg override, sau đó config)
    hf_cache = hf_cache_override or paths.hf_cache
    if hf_cache:
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
        os.environ["HF_HUB_CACHE"] = str(hf_cache)

    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split=split,
        trust_remote_code=True,
        verification_mode="no_checks",
        cache_dir=str(hf_cache) if hf_cache else None,
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    count_samples = 0
    count_skipped_pragma = 0

    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            if idx < start_idx:
                continue
            if idx >= end_idx:
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
            if not sol_path.exists():
                sol_path.write_text(src, encoding="utf-8")

            report_path = REPORT_DIR / f"contract_{idx}.json"
            # chạy slither nếu chưa có report
            if not report_path.exists():
                run_slither_once(sol_path, report_path)

            if not report_path.exists():
                continue

            vuln_map_all = parse_slither_json(report_path)
            if not vuln_map_all:
                continue

            # map đúng file key trong report
            keys = [str(sol_path), sol_path.name, f"contracts/contract_{idx}.sol"]
            vuln_map = None
            for k in keys:
                if k in vuln_map_all:
                    vuln_map = vuln_map_all[k]
                    break
            if vuln_map is None:
                first_file = next(iter(vuln_map_all.keys()))
                vuln_map = vuln_map_all[first_file]

            for func_name, label in vuln_map.items():
                if label not in TARGET_LABELS:
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
                jsonl_append(fout, {
                    "orig_id": idx,
                    "file": str(sol_path),
                    "function": func_name,
                    "label": label,
                    "clean_code": clean,
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                })
                count_samples += 1

    print("DONE.")
    print("samples ghi được:", count_samples)
    print("bỏ vì pragma:", count_skipped_pragma)
