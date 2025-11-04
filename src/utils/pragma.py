import re
PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)

TARGET = (0, 8, 17)

def parse_version_token(tok: str):
    # bỏ ký tự so sánh để lấy x.y.z
    for pre in ("^", "=", ">=", "<=", ">", "<"):
        if tok.startswith(pre):
            tok = tok[len(pre):].strip()
            break
    parts = tok.split(".")
    parts += ["0"] * (3 - len(parts))
    try:
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return None

def compatible_with_0_8_17(pragma_raw: str) -> bool:
    # chấp nhận các tổ hợp tương thích 0.8.x; loại <0.8.0
    toks = pragma_raw.replace("&&"," ").split()
    seen_ok = False
    for t in toks:
        t = t.strip()
        if not t: 
            continue
        if t[0].isdigit() or t.startswith(("^","=")):
            v = parse_version_token(t)
            if not v: return False
            if v[0] == 0 and v[1] == 8:
                seen_ok = True
            else:
                return False
        elif t.startswith(">="):
            v = parse_version_token(t); 
            if not v: return False
            if (v[0], v[1]) < (0,8): return False
            seen_ok = True
        elif t.startswith("<") or t.startswith("<="):
            v = parse_version_token(t)
            if not v: return False
            # thường <0.9.0 là OK; <0.8.0 là loại
            if (v[0], v[1]) < (0,8): return False
        else:
            return False
    return seen_ok

def first_pragma(src: str):
    m = PRAGMA_RE.search(src)
    return m.group(1).strip() if m else None
