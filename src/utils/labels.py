# src/utils/labels.py
TARGET_LABELS = {
    "reentrancy",
    "timestamp_dependency",
    "unchecked_call",
    "tx_origin_misuse",
}

def normalize_label(name: str) -> str:
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
