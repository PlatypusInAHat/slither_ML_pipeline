from pathlib import Path
import json

def jsonl_reader(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
