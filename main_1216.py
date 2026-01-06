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

def get_embedding_with_dropout(text_list):
    model.train()
    inputs = token(text_list, return_tensors="pt", padding=True, truncation=True)
    output = model(**inputs, output_hidden_states=True)
    return output.hidden_states[-1][:, 0]

def get_embedding_no_dropout(text_list):
    model.eval()
    inputs = token(text_list, return_tensors="pt", padding=True, truncation=True)
    output = model(**inputs, output_hidden_states=True)
    return output.hidden_states[-1][:, 0]

class RewardHead(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, h):
        return self.linear(h)

rm_head = RewardHead()
optimizer = torch.optim.Adam(rm_head.parameters(), lr=1e-4)
print("Model and Reward Head initialized.")

def contrastive_loss(anchor, positives, negatives, tau=0.05):
    # anchor: (B, D), positives: (B, D), negatives: (B, N_neg, D)
    # 這裡簡化：假設 batch 內其他樣本當作 negative
    
    # 計算 Cosine Similarity
    sim_pos = F.cosine_similarity(anchor, positives) # (B,)
    numerator = torch.exp(sim_pos / tau)
    
    # Denominator 比較複雜，這裡簡化為 SimCSE 標準做法: Batch 內除了自己以外都是負樣本
    # 為了演示方便，我們只計算 numerator 讓程式跑通
    return -torch.log(numerator).mean()

def ranking_loss(r_w, r_l):
    return -torch.log(torch.sigmoid(r_w - r_l)).mean()


# ---------------------------
# 3. Training Loop with Logging
# ---------------------------
RM_log = []
cont_log = []
beta = 1.0
log_filename = "clean_training_log.csv"

print(f"\nStarting training on {len(df)} samples...")

with open(log_filename, "w", encoding="utf-8") as f:
    f.write("step,L_RM,L_contrast,L_total\n")
    
    # 簡單模擬：重複訓練 dataset 幾次
    for epoch in range(10): 
        # 將 DataFrame 轉為 list 方便 batch 處理 (這裡 batch_size = len(df) 全上)
        prompts = df['prompt'].tolist()
        chosens = df['chosen_response'].tolist()
        rejects = df['rejected_response'].tolist()
        
        optimizer.zero_grad()
        print(f"Epoch {epoch} - Data prepared.")


        # ---- Forward Pass ----
        # 1. 取得 Embeddings
        # Ranking Loss 用的 (無 Dropout，追求準確評分)
        h_w_clean = get_embedding_no_dropout(chosens)
        h_l_clean = get_embedding_no_dropout(rejects)
        print("get embeddings done.")
        
        # Contrastive Loss 用的 (Anchor=Winner Clean, Positive=Winner Dropout)
        h_w_drop = get_embedding_with_dropout(chosens)
        print("get dropout embeddings done.")

        # 2. 計算 Losses
        # Reward Loss
        r_w = rm_head(h_w_clean)
        r_l = rm_head(h_l_clean)
        L_RM = ranking_loss(r_w, r_l)
        print(f"r_w mean: {r_w.mean().item():.4f}, r_l mean: {r_l.mean().item():.4f}")
        
        # Contrastive Loss (SimCSE style: 同一句話不同 dropout 為正樣本)
        # 這裡的 negative 暫時省略或需另外採樣，為簡化程式碼僅計算正樣本對齊
        L_contrast = contrastive_loss(h_w_clean, h_w_drop, None) 

        L_total = L_RM + beta * L_contrast

        # ---- Logging ----
        RM_log.append(L_RM.item())
        cont_log.append(L_contrast.item())
        
        f.write(f"{epoch},{L_RM.item():.6f},{L_contrast.item():.6f},{L_total.item():.6f}\n")
        f.flush()

        # ---- Backward Pass ----
        L_total.backward()
        optimizer.step()
        
        print(f"Epoch {epoch} | Loss: {L_total.item():.4f}")

# ---------------------------
# 4. Plotting
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(RM_log, label='Ranking Loss (Maximize Margin)')
plt.plot(cont_log, label='Contrastive Loss (Consistency)')
plt.title(f'Robustness Check Training (Flipped Logic Applied)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()