# src/features/clean.py
import re

def strip_comments_and_whitespace(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//.*", "", src)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return " ".join(lines)

def extract_function_source(code: str, func_name: str) -> str:
    pat = re.compile(r"\bfunction\s+" + re.escape(func_name) + r"\s*\([^)]*\)\s*[^{;]*\{", re.DOTALL)
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
