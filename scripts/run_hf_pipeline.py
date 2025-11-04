# scripts/run_hf_pipeline.py
import argparse
from src.builders.hf_pipeline import build_from_hf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths", default="configs/paths.windows.yaml")
    p.add_argument("--labels", default="configs/labels.yaml")
    p.add_argument("--start_idx", type=int, default=39826)
    p.add_argument("--end_idx", type=int, default=200000)      # duyệt tới < end
    p.add_argument("--hf_cache", default=None)                 # override nếu muốn
    p.add_argument("--split", default="train")
    p.add_argument("--config", default="big-multilabel")
    p.add_argument("--dataset", default="mwritescode/slither-audited-smart-contracts")
    args = p.parse_args()

    build_from_hf(
        paths_yaml=args.paths,
        labels_yaml=args.labels,
        dataset_id=args.dataset,
        dataset_config=args.config,
        split=args.split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        hf_cache_override=args.hf_cache,
    )

if __name__ == "__main__":
    main()
