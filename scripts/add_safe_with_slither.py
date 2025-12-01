import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Set, Tuple, List, Any

from transformers import AutoTokenizer


# ====== CẤU HÌNH ======

# Gốc repo (chỉnh lại nếu repo ở chỗ khác)
REPO_ROOT = Path(r"D:\slither-ml-pipeline").resolve()

# File JSONL gốc (vuln-only, build từ hf_pipeline)
IN_JSONL = REPO_ROOT / "data" / "processed" / "dataset_from_hf.jsonl"

# File JSONL mới (vuln + safe)
OUT_JSONL = REPO_ROOT / "data" / "processed" / "slither_big_multilabel_with_safe.jsonl"

# Thư mục chứa report Slither (dùng lại / tạo mới đều được)
REPORT_DIR = REPO_ROOT / "data" / "interim" / "slither_safe" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Giới hạn token cho CodeBERT
MAX_TOKENS = 510

SAFE_LABEL = "safe"

# Regex tìm function name
FUNC_DEF_RE = re.compile(r"\bfunction\s+([A-Za-z_]\w*)\s*\(")


# ====== HÀM PHỤ ======

def strip_comments_and_whitespace(src: str) -> str:
    """Loại bỏ comment /* */ và //, gom code về 1 dòng."""
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//.*", "", src)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return " ".join(lines)


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


def parse_orig_id_from_path(path_str: str) -> int | None:
    """
    Thử lấy orig_id từ tên file contract_X.sol.
    Nếu không parse được thì trả None.
    """
    m = re.search(r"contract_(\d+)\.sol", path_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_slither_all_issues(report_path: Path) -> Dict[str, Set[str]]:
    """
    Đọc report Slither JSON và trả về:
    {
      "filename.sol": { "func1", "func2", ... },
      ...
    }
    => Mọi hàm có bất kỳ detector nào, không filter 4 label nữa.
    """
    with report_path.open("r", encoding="utf-8") as f:
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


# ====== MAIN ======

def main():
    print(f"Đọc dataset gốc từ: {IN_JSONL}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    contract_to_vuln_funcs: Dict[str, Set[str]] = {}
    seen_pairs: Set[Tuple[str, str]] = set()
    original_lines: List[str] = []

    # 1) Đọc JSONL gốc, build map và lưu các dòng cũ
    with IN_JSONL.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            original_lines.append(line)

            obj = json.loads(line)
            file_path = obj.get("file")
            func_name = obj.get("function")

            if not file_path or not func_name:
                continue

            seen_pairs.add((file_path, func_name))
            contract_to_vuln_funcs.setdefault(file_path, set()).add(func_name)

    print(f"Số dòng vuln ban đầu: {len(original_lines)}")
    print(f"Số contract trong JSONL: {len(contract_to_vuln_funcs)}")

    total_safe_written = 0

    # 2) Mở file output NGAY BÂY GIỜ và ghi luôn các dòng cũ
    print(f"Ghi output mới vào: {OUT_JSONL}")
    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        # ghi toàn bộ sample vuln cũ
        for line in original_lines:
            fout.write(line + "\n")

        # 3) Cho mỗi contract, chạy Slither để biết hàm nào có bất kỳ lỗi nào,
        #    rồi vừa tìm được safe là ghi luôn
        for file_path, vuln_funcs in contract_to_vuln_funcs.items():
            sol_path = Path(file_path)

            # Nếu path là relative, nối với REPO_ROOT
            if not sol_path.is_absolute():
                sol_path = (REPO_ROOT / sol_path).resolve()

            if not sol_path.exists():
                print(f"[WARN] Không tìm thấy file Solidity: {sol_path}")
                continue

            # Đọc source
            try:
                src = sol_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[WARN] Lỗi đọc {sol_path}: {e}")
                continue

            # Report Slither cho file này (file JSON trong thư mục REPORT_DIR)
            report_name = sol_path.stem + ".json"
            report_path = REPORT_DIR / report_name

            if not report_path.exists():
                print(f"[INFO] Chạy Slither cho {sol_path} ...")

                # suy ra root_dir cho remap "slither_hf"
                try:
                    root_dir = sol_path.parent.parent
                except Exception:
                    root_dir = sol_path.parent

                remap = f"slither_hf={root_dir}"

                cmd = [
                    "python",
                    "-m",
                    "slither",
                    str(sol_path),
                    "--solc-remaps",
                    remap,
                    "--json",
                    str(report_path),
                ]

                try:
                    subprocess.run(cmd, check=False, cwd=root_dir, timeout=300)
                except subprocess.TimeoutExpired:
                    print(f"[WARN] Slither timeout cho {sol_path}, bỏ qua safe.")
                    continue

            if not report_path.exists():
                print(f"[WARN] Slither không tạo report cho {sol_path}, bỏ qua safe.")
                continue

            # Parse mọi issue Slither
            issues_map = parse_slither_all_issues(report_path)

            # Map key giống như trong pipeline: thử 3 key
            keys = [
                str(sol_path),
                sol_path.name,
                f"contracts/{sol_path.name}",
            ]
            funcs_with_any_issue: Set[str] = set()
            for k in keys:
                if k in issues_map:
                    funcs_with_any_issue |= issues_map[k]

            # Liệt kê tất cả function trong contract
            all_funcs = list_all_functions(src)

            # orig_id: cố lấy từ tên file contract_X.sol
            orig_id = parse_orig_id_from_path(str(sol_path))

            safe_count_here = 0

            for func_name in all_funcs:
                pair = (file_path, func_name)

                # Nếu đã có trong JSONL (vuln) → giữ nguyên, không ghi thêm safe
                if pair in seen_pairs:
                    continue

                # Nếu hàm có bất kỳ issue nào theo Slither → KHÔNG safe
                if func_name in funcs_with_any_issue:
                    continue

                # Ngược lại → safe
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

                obj: Dict[str, Any] = {
                    "orig_id": orig_id,
                    "file": str(file_path),
                    "function": func_name,
                    "label": SAFE_LABEL,
                    "clean_code": clean,
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }

                # ghi TRỰC TIẾP từng safe sample
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_safe_written += 1
                safe_count_here += 1

            print(
                f"[INFO] {sol_path.name}: {len(all_funcs)} hàm, "
                f"{len(vuln_funcs)} vuln theo JSONL, "
                f"{len(funcs_with_any_issue)} hàm có issue theo Slither, "
                f"{total_safe_written} safe đã ghi."
            )

    print("DONE.")
    print(f"Tổng dòng vuln cũ: {len(original_lines)}")
    print(f"Tổng dòng safe thêm vào: {total_safe_written}")
    print(f"Tổng dòng trong file mới: {len(original_lines) + total_safe_written}")


if __name__ == "__main__":
    main()
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Set, Tuple, List, Any

from transformers import AutoTokenizer


# ====== CẤU HÌNH ======

# Gốc repo (chỉnh lại nếu repo ở chỗ khác)
REPO_ROOT = Path(r"D:\slither-ml-pipeline").resolve()

# File JSONL gốc (vuln-only, build từ hf_pipeline)
IN_JSONL = REPO_ROOT / "data" / "processed" / "dataset_from_hf.jsonl"

# File JSONL mới (vuln + safe)
OUT_JSONL = REPO_ROOT / "data" / "processed" / "slither_big_multilabel_with_safe.jsonl"

# Thư mục chứa report Slither (dùng lại / tạo mới đều được)
REPORT_DIR = REPO_ROOT / "data" / "interim" / "slither_safe" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Giới hạn token cho CodeBERT
MAX_TOKENS = 510

SAFE_LABEL = "safe"

# Regex tìm function name
FUNC_DEF_RE = re.compile(r"\bfunction\s+([A-Za-z_]\w*)\s*\(")


# ====== HÀM PHỤ ======

def strip_comments_and_whitespace(src: str) -> str:
    """Loại bỏ comment /* */ và //, gom code về 1 dòng."""
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//.*", "", src)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return " ".join(lines)


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


def parse_orig_id_from_path(path_str: str) -> int | None:
    """
    Thử lấy orig_id từ tên file contract_X.sol.
    Nếu không parse được thì trả None.
    """
    m = re.search(r"contract_(\d+)\.sol", path_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_slither_all_issues(report_path: Path) -> Dict[str, Set[str]]:
    """
    Đọc report Slither JSON và trả về:
    {
      "filename.sol": { "func1", "func2", ... },
      ...
    }
    => Mọi hàm có bất kỳ detector nào, không filter 4 label nữa.
    """
    with report_path.open("r", encoding="utf-8") as f:
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


# ====== MAIN ======

def main():
    print(f"Đọc dataset gốc từ: {IN_JSONL}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    contract_to_vuln_funcs: Dict[str, Set[str]] = {}
    seen_pairs: Set[Tuple[str, str]] = set()
    original_lines: List[str] = []

    # 1) Đọc JSONL gốc, build map và lưu các dòng cũ
    with IN_JSONL.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            original_lines.append(line)

            obj = json.loads(line)
            file_path = obj.get("file")
            func_name = obj.get("function")

            if not file_path or not func_name:
                continue

            seen_pairs.add((file_path, func_name))
            contract_to_vuln_funcs.setdefault(file_path, set()).add(func_name)

    print(f"Số dòng vuln ban đầu: {len(original_lines)}")
    print(f"Số contract trong JSONL: {len(contract_to_vuln_funcs)}")

    safe_samples: List[str] = []

    # 2) Cho mỗi contract, chạy Slither để biết hàm nào có bất kỳ lỗi nào
    for file_path, vuln_funcs in contract_to_vuln_funcs.items():
        sol_path = Path(file_path)

        # Nếu path là relative, nối với REPO_ROOT
        if not sol_path.is_absolute():
            sol_path = (REPO_ROOT / sol_path).resolve()

        if not sol_path.exists():
            print(f"[WARN] Không tìm thấy file Solidity: {sol_path}")
            continue

        # Đọc source
        try:
            src = sol_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Lỗi đọc {sol_path}: {e}")
            continue

        # Report Slither cho file này (file JSON trong thư mục REPORT_DIR)
        report_name = sol_path.stem + ".json"
        report_path = REPORT_DIR / report_name

        if not report_path.exists():
            print(f"[INFO] Chạy Slither cho {sol_path} ...")

            # suy ra root_dir cho remap "slither_hf"
            # ví dụ: D:\slither_hf\contracts\contract_480.sol
            # => root_dir = D:\slither_hf
            try:
                root_dir = sol_path.parent.parent
            except Exception:
                root_dir = sol_path.parent

            remap = f"slither_hf={root_dir}"

            cmd = [
                "python",
                "-m",
                "slither",
                str(sol_path),
                "--solc-remaps",
                remap,
                "--json",
                str(report_path),
            ]

            try:
                # đặt cwd = root_dir + timeout để khỏi treo vô hạn
                subprocess.run(cmd, check=False, cwd=root_dir, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"[WARN] Slither timeout cho {sol_path}, bỏ qua safe.")
                continue

        if not report_path.exists():
            print(f"[WARN] Slither không tạo report cho {sol_path}, bỏ qua safe.")
            continue

        # Parse mọi issue Slither
        issues_map = parse_slither_all_issues(report_path)

        # Map key giống như trong pipeline: thử 3 key
        keys = [
            str(sol_path),
            sol_path.name,
            f"contracts/{sol_path.name}",
        ]
        funcs_with_any_issue: Set[str] = set()
        for k in keys:
            if k in issues_map:
                funcs_with_any_issue |= issues_map[k]

        # Liệt kê tất cả function trong contract
        all_funcs = list_all_functions(src)

        # orig_id: cố lấy từ tên file contract_X.sol
        orig_id = parse_orig_id_from_path(str(sol_path))
        # nếu không parse được thì để None, vẫn ghi được

        for func_name in all_funcs:
            pair = (file_path, func_name)

            # Nếu đã có trong JSONL (vuln) → giữ nguyên, không ghi thêm safe
            if pair in seen_pairs:
                continue

            # Nếu hàm có bất kỳ issue nào theo Slither → KHÔNG safe
            if func_name in funcs_with_any_issue:
                continue

            # Ngược lại → safe
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

            obj: Dict[str, Any] = {
                "orig_id": orig_id,
                "file": str(file_path),
                "function": func_name,
                "label": SAFE_LABEL,
                "clean_code": clean,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
            safe_samples.append(json.dumps(obj, ensure_ascii=False))

        print(
            f"[INFO] {sol_path.name}: {len(all_funcs)} hàm, "
            f"{len(vuln_funcs)} vuln theo JSONL, "
            f"{len(funcs_with_any_issue)} hàm có issue theo Slither, "
            f"{len(safe_samples)} safe tích lũy."
        )

    # 3) Ghi file mới: tất cả dòng cũ + safe
    print(f"Ghi output mới vào: {OUT_JSONL}")
    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        for line in original_lines:
            fout.write(line + "\n")
        for line in safe_samples:
            fout.write(line + "\n")

    print("DONE.")
    print(f"Tổng dòng vuln cũ: {len(original_lines)}")
    print(f"Tổng dòng safe thêm vào: {len(safe_samples)}")
    print(f"Tổng dòng trong file mới: {len(original_lines) + len(safe_samples)}")


if __name__ == "__main__":
    main()
