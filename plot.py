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
# 8. Plot
# ---------------------------
df_log = pd.read_csv("training_log_imptt.csv")

plt.figure(figsize=(10, 6))

target_exp = "E1_Poison_Contrastive"
sub = df_log[df_log["exp"] == target_exp]

if not sub.empty:
    plt.plot(sub["step"], sub["L_RM"], label=f"{target_exp} (L_RM)")
    plt.plot(sub["step"], sub["L_cont"], label=f"{target_exp} (L_cont)")
    # plt.plot(sub["step"], sub["L_total"], 
    # label=f"{target_exp} (L_total)")

# for exp in df_log["exp"].unique():
#     if df_log["exp"] == "E1_Poison_Contrastive":
#         sub = df_log[df_log["exp"] == exp]
#         plt.plot(sub["step"], sub["L_RM"], label=exp)
#         plt.plot(sub["step"], sub["L_cont"], label=exp)

plt.xlabel("Training Step")
plt.ylabel("Total Loss")
plt.title("Reward Model Training (Poison, RM loss + Contrastive loss)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("Poison RM Loss")
