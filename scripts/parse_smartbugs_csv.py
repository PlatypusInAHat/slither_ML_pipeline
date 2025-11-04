import pandas as pd
from pathlib import Path
from src.utils.labels import normalize_smartbugs_label

INP = Path("data/raw/smartbugs_wild.csv")
OUT = Path("data/processed/smartbugs_wild.parquet")

df = pd.read_csv(INP)
# chuẩn hoá label theo 4 nhóm trọng tâm
df["label_canonical"] = df["label"].map(normalize_smartbugs_label)
df.to_parquet(OUT, index=False)
print("Parquet ->", OUT)
