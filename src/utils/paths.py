# src/utils/paths.py
from dataclasses import dataclass
import yaml

@dataclass
class PathsConfig:
    base_dir: str
    hf_cache: str | None = None

    @staticmethod
    def from_yaml(path: str) -> "PathsConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return PathsConfig(
            base_dir=d["BASE_DIR"],
            hf_cache=d.get("HF_CACHE"),
        )
