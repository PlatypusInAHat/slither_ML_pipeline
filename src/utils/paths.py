from pathlib import Path
import yaml, os

def load_yaml(p): 
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_project_root():
    return Path(os.getcwd())  # hoặc cố định nếu muốn

