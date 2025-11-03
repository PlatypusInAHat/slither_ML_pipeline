from pathlib import Path
import json, csv

INP = Path("data/processed/dataset_from_hf.jsonl")
OUT = Path("data/processed/dataset_from_hf.csv")

fields = ["orig_id","file","function","label","clean_code"]
with INP.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8", newline="") as fout:
    w = csv.DictWriter(fout, fieldnames=fields); w.writeheader()
    for line in fin:
        obj = json.loads(line)
        w.writerow({k: obj.get(k,"") for k in fields})
print("CSV ->", OUT)
