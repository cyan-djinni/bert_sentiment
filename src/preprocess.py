# src/preprocess.py

import pandas as pd
import re
import os

DATA_DIR = "data"
SPLITS = ["Train", "Valid", "Test"]

def clean_text(text):
    """
    轻量清理文本：
    - 去掉前后空格
    - 去掉换行符和制表符
    - 多空格合并为一个空格
    """
    text = str(text).strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text

for split in SPLITS:
    file_path = os.path.join(DATA_DIR, f"{split}.csv")
    df = pd.read_csv(file_path)

    # 清理文本
    df["text"] = df["text"].apply(clean_text)

    # 打印信息
    print(f"=== {split.upper()} ===")
    print(f"Total samples: {len(df)}")
    if "label" in df.columns:
        print("Label distribution:")
        print(df["label"].value_counts())
    print("-" * 30)

    # 可选：保存清理后的 CSV
    clean_file_path = os.path.join(DATA_DIR, f"{split}_clean.csv")
    df.to_csv(clean_file_path, index=False)