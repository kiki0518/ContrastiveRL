import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import pandas as pd
import io

csv_filename = "pku_saferlhf_flip_harmfuleness_gap.csv"

import os

if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename, encoding="utf-8")
else:
    raise FileNotFoundError(f"CSV file not found: {csv_filename}")

print("--- Original Data Head ---")
print(df[['prompt', 'flipped']].head())

df = df.fillna("")
df['chosen_response'] = df['chosen_response'].astype(str)
df['rejected_response'] = df['rejected_response'].astype(str)
df['prompt'] = df['prompt'].astype(str)

print(f"清洗後資料量: {len(df)} 筆")


# ---------------------------
# 2. Model & Loss Setup
# ---------------------------
token = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Model and Tokenizer loaded.")
