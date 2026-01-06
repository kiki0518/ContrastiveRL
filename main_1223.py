import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 0. Device (Mac Safe)
# ---------------------------
device = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ---------------------------
# 1. Load Data
# ---------------------------
csv_filename = "pku_saferlhf_flip_harmfuleness_gap.csv"

if not os.path.exists(csv_filename):
    raise FileNotFoundError(csv_filename)

df = pd.read_csv(csv_filename, encoding="utf-8")
df = df.fillna("").astype(str)

print(f"Dataset size: {len(df)}")

# ---------------------------
# 2. Model Setup
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
).to(device)

# 🔒 Freeze backbone (CRITICAL)
for p in model.parameters():
    p.requires_grad = False

print("Backbone frozen.")

class RewardHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x)

rm_head = RewardHead().to(device)
optimizer = torch.optim.Adam(rm_head.parameters(), lr=1e-4)

# ---------------------------
# 3. Embedding Function
# ---------------------------
def get_embedding(texts, train=False):
    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0]  # CLS

# ---------------------------
# 4. Losses
# ---------------------------
def ranking_loss(r_w, r_l):
    return -torch.log(torch.sigmoid(r_w - r_l)).mean()

def contrastive_loss(h1, h2, tau=0.05):
    sim = F.cosine_similarity(h1, h2)
    return -torch.log(torch.exp(sim / tau)).mean()

# ---------------------------
# 5. Training Loop (Mini-batch)
# ---------------------------
BATCH_SIZE = 16   # Mac safe
EPOCHS = 5
beta = 1.0

RM_log, cont_log = [], []

log_file = open("clean_training_log.csv", "w", encoding="utf-8")
log_file.write("step,L_RM,L_contrast,L_total\n")

step = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch}")

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]

        chosen = batch["chosen_response"].tolist()
        rejected = batch["rejected_response"].tolist()

        optimizer.zero_grad()

        # ----- Embeddings -----
        h_w_clean = get_embedding(chosen, train=False)
        h_l_clean = get_embedding(rejected, train=False)
        h_w_drop  = get_embedding(chosen, train=True)
        h_l_drop = get_embedding(rejected, train=True)

        # ----- Reward -----
        r_w = rm_head(h_w_clean)
        r_l = rm_head(h_l_clean)
        L_RM = ranking_loss(r_w, r_l)

        # ----- Contrastive -----
        # L_cont = contrastive_loss(h_w_clean.detach(), h_w_drop)

        L_cont_w = contrastive_loss(
            h_w_clean.detach(),
            h_w_drop
        )

        L_cont_l = contrastive_loss(
            h_l_clean.detach(),
            h_l_drop
        )

        L_cont = 0.5 * (L_cont_w + L_cont_l)


        L_total = L_RM + beta * L_cont
        L_total.backward()
        optimizer.step()

        RM_log.append(L_RM.item())
        cont_log.append(L_cont.item())

        log_file.write(
            f"{step},{L_RM.item():.6f},{L_cont.item():.6f},{L_total.item():.6f}\n"
        )

        if step % 200 == 0:
            print(
                f"Step {step} | "
                f"RM: {L_RM.item():.4f} | "
                f"Cont: {L_cont.item():.4f}"
            )

        step += 1

log_file.close()

# ---------------------------
# 6. Plot
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(RM_log, label="Ranking Loss")
plt.plot(cont_log, label="Contrastive Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Mac-safe Reward + Contrastive Training")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
