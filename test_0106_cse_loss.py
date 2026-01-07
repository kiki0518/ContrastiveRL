# ============================================================
# Reward Model + Contrastive Robustness (Clean vs Poison)
# ============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoTokenizer, AutoModel

# ---------------------------
# 0. Device
# ---------------------------
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ---------------------------
# 1. Load Data
# ---------------------------
CSV_FILE = "pku_saferlhf_flip_harmfuleness_gap.csv"
assert os.path.exists(CSV_FILE)

df = pd.read_csv(CSV_FILE).fillna("").astype(str)
df["flipped"] = df["flipped"].astype(int)

df_clean  = df[df["flipped"] == 0].reset_index(drop=True)
df_poison = df[df["flipped"] == 1].reset_index(drop=True)

print(f"Clean: {len(df_clean)} | Poison: {len(df_poison)}")

# ---------------------------
# 2. Model & Head
# ---------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class RewardHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x)

# ---------------------------
# 3. Losses
# ---------------------------
def ranking_loss(r_w, r_l):
    return -torch.log(torch.sigmoid(r_w - r_l)).mean()

def contrastive_loss(h_ref, h_cur, tau=0.05):
    sim = F.cosine_similarity(h_ref, h_cur)
    return -torch.log(torch.exp(sim / tau)).mean()

def simcse_loss(h1, h2, tau=0.05):
    # h1, h2 shape: [batch_size, hidden_dim]
    sim_matrix = F.cosine_similarity(h1.unsqueeze(1), h2.unsqueeze(0), dim=2) # [B, B]
    sim_matrix = sim_matrix / tau
    
    # 對角線是正樣本對 (Positive Pairs)
    labels = torch.arange(h1.size(0)).to(h1.device)
    return F.cross_entropy(sim_matrix, labels)

# ---------------------------
# 4. Embedding
# ---------------------------
def get_embedding(model, texts, train):
    model.train() if train else model.eval()
    with torch.set_grad_enabled(train):
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        out = model(**inputs)
        return out.last_hidden_state[:, 0]

# ---------------------------
# 5. Logger
# ---------------------------
class CSVLogger:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.f.write(
            "exp,epoch,step,data_type,objective,L_RM,L_cont,L_total\n"
        )
        self.f.flush()

    def log(self, exp, epoch, step, data_type, objective,
            L_RM, L_cont, L_total):
        self.f.write(
            f"{exp},{epoch},{step},{data_type},{objective},"
            f"{L_RM:.6f},{L_cont:.6f},{L_total:.6f}\n"
        )
        self.f.flush()

    def close(self):
        self.f.close()

# ---------------------------
# 6. Training (Modified)
# ---------------------------
def train_experiment(
    df, exp_name, data_type, use_contrastive, logger,
    epochs=3, batch_size=32, beta=1.0, force_clean=False,
    dropout_rate=0.05  
):
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    config.hidden_dropout_prob = dropout_rate
    config.attention_probs_dropout_prob = dropout_rate
    
    model = AutoModel.from_pretrained(MODEL_NAME, config=config).to(device)
    rm_head = RewardHead().to(device)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-5},
        {"params": rm_head.parameters(), "lr": 1e-4},
    ])

    step = 0

    for epoch in range(epochs):
        # 打亂數據以增加隨機性
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(df_shuffled), batch_size):
            batch = df_shuffled.iloc[i:i+batch_size]
            
            chosen = []
            rejected = []
            
            # 處理標籤邏輯
            for _, row in batch.iterrows():
                c, r = row["chosen_response"], row["rejected_response"]
                # 如果要跑 Clean 實驗且該樣本是被 flip 過的，則換回來
                if force_clean and int(row["flipped"]) == 1:
                    chosen.append(r)
                    rejected.append(c)
                else:
                    chosen.append(c)
                    rejected.append(r)

            optimizer.zero_grad()

            h_w_1 = get_embedding(model, chosen, train=True)
            h_l_1 = get_embedding(model, rejected, train=True)

            r_w = rm_head(h_w_1)
            r_l = rm_head(h_l_1)
            L_RM = ranking_loss(r_w, r_l)

            if use_contrastive:
                # 對比損失通常針對 Embedding 空間，這裡採用的策略是保持 representation 穩定
                h_w_2 = get_embedding(model, chosen, train=True)
                h_l_2 = get_embedding(model, rejected, train=True)

                h1_all = torch.cat([h_w_1, h_l_1], dim=0) 
                h2_all = torch.cat([h_w_2, h_l_2], dim=0)

                h1_combined = torch.cat([h_w_1, h_l_1], dim=0)
                h2_combined = torch.cat([h_w_2, h_l_2], dim=0)

                L_cont = simcse_loss(h1_combined, h2_combined, tau=0.05)

            else:
                L_cont = torch.tensor(0.0, device=device)

            L_total = L_RM + beta * L_cont
            L_total.backward()
            optimizer.step()

            logger.log(
                exp_name, epoch, step, data_type,
                "rm+cont" if use_contrastive else "rm",
                L_RM.item(), L_cont.item(), L_total.item()
            )

            if step % 200 == 0:
                print(
                    f"[{exp_name}] step {step} | RM {L_RM.item():.4f} | Cont {L_cont.item():.4f}"
                )
            step += 1
# ---------------------------
# 7. Run Experiments (Updated)
# ---------------------------
logger = CSVLogger("training_log.csv")

# 這裡我們使用完整的 df，不再分開 filter
experiments = [
    # 1. Poison 實驗：不修復標籤 (force_clean=False)，直接跑被污染的資料
    # ("E1_Poison_Contrastive", df, "poison", True,  False),
    # ("E2_Poison_RM",          df, "poison", False, False),
    
    # 2. Clean 實驗：遇到 flipped=1 的要修正回來 (force_clean=True)
    ("E3_Clean_Contrastive",  df, "clean",  True,  True),
    ("E4_Clean_RM",           df, "clean",  False, True),
]

for exp, df_exp, dtype, use_cont, should_fix in experiments:
    print(f"\n===== {exp} (Fix Labels: {should_fix}) =====")
    train_experiment(
        df=df_exp,
        exp_name=exp,
        data_type=dtype,
        use_contrastive=use_cont,
        logger=logger,
        force_clean=should_fix  # 傳入修復開關
    )

logger.close()

print("\n✅ Training finished. Log saved.")

# ---------------------------
# 8. Plot
# ---------------------------
df_log = pd.read_csv("training_log.csv")

plt.figure(figsize=(10, 6))
for exp in df_log["exp"].unique():
    sub = df_log[df_log["exp"] == exp]
    plt.plot(sub["step"], sub["L_total"], label=exp)

plt.xlabel("Training Step")
plt.ylabel("Total Loss")
plt.title("Reward Model Training (Clean vs Poison)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
