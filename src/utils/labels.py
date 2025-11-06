import yaml

def load_labels(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    label2id = y["labels"]
    id2label = {v:k for k,v in label2id.items()}
    return label2id, id2label
