.PHONY: setup hf jsonl2csv

setup:
\tpython -m pip install -r requirements.txt

hf:
\tpython scripts/run_hf_pipeline.py

jsonl2csv:
\tpython scripts/convert_jsonl_to_csv.py
