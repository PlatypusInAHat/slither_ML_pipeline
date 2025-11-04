# src/utils/io.py
import json

def jsonl_append(fout, obj):
    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
