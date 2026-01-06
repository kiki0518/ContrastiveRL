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
BATCH_SIZE = 16   
EPOCHS = 5
beta = 1.0

RM_log, cont_log = [], []

log_file = open("clean_training_log.csv", "w", encoding="utf-8")
log_file.write("step,L_RM,L_contrast,L_total\n")

step = 0

def train_reward_model(df, use_contrastive, epochs=3):
    rm_head = RewardHead().to(device)
    optimizer = torch.optim.Adam(list(rm_head.parameters()) + list(model.parameters()), lr=1e-5)

    for epoch in range(epochs):
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]

            chosen = batch["chosen_response"].tolist()
            rejected = batch["rejected_response"].tolist()

            optimizer.zero_grad()

            # clean embeddings
            h_w = get_embedding(chosen, train=False)
            h_l = get_embedding(rejected, train=False)

            # ranking loss
            r_w = rm_head(h_w)
            r_l = rm_head(h_l)
            L_RM = ranking_loss(r_w, r_l)

            if use_contrastive:
                h_w_d = get_embedding(chosen, train=True)
                h_l_d = get_embedding(rejected, train=True)

                L_cont = 0.5 * (
                    contrastive_loss(h_w.detach(), h_w_d) +
                    contrastive_loss(h_l.detach(), h_l_d)
                )
            else:
                L_cont = 0.0

            L_total = L_RM + beta * L_cont
            L_total.backward()
            optimizer.step()

    return rm_head



def evaluate_robustness(df, rm_head):
    deltas = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]

        chosen = batch["chosen_response"].tolist()
        rejected = batch["rejected_response"].tolist()

        # clean
        h_w = get_embedding(chosen, train=False)
        h_l = get_embedding(rejected, train=False)
        margin_clean = (rm_head(h_w) - rm_head(h_l)).detach()

        # perturbed (dropout)
        h_w_d = get_embedding(chosen, train=True)
        h_l_d = get_embedding(rejected, train=True)
        margin_drop = (rm_head(h_w_d) - rm_head(h_l_d)).detach()

        delta = torch.abs(margin_clean - margin_drop)
        deltas.append(delta.cpu())

    deltas = torch.cat(deltas)
    return {
        "mean_delta": deltas.mean().item(),
        "std_delta": deltas.std().item()
    }

rm_plain = train_reward_model(df, use_contrastive=False)
rm_cont  = train_reward_model(df, use_contrastive=True)

plain_stats = evaluate_robustness(df, rm_plain)
cont_stats  = evaluate_robustness(df, rm_cont)

print("RM only:", plain_stats)
print("RM + Contrastive:", cont_stats)


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
