from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------
# Load backbone
# ---------------------------
token = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------
# Helper: get CLS embedding
# ---------------------------
def get_embedding_no_dropout(text):
    model.eval()              # 關閉 dropout
    inputs = token(text, return_tensors="pt")
    # with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)
    return output.hidden_states[-1][:, 0]   # CLS embedding


def get_embedding_with_dropout(text):
    model.train()             # 啟用 dropout
    inputs = token(text, return_tensors="pt")
    # with torch.no_grad():     # 不更新 backbone
    output = model(**inputs, output_hidden_states=True)
    return output.hidden_states[-1][:, 0]



# ---------------------------
# Contrastive loss (SimCSE)
# ---------------------------
def sim(a, b):
    return F.cosine_similarity(a, b)

def contrastive_loss(anchor, positives, negatives, tau=0.05):
    # numerator
    pos = torch.exp(sim(anchor, positives) / tau)

    # denominator
    all_samples = torch.cat([positives.unsqueeze(0), negatives])
    denom = torch.exp(sim(anchor, all_samples) / tau).sum()

    return -torch.log(pos / denom)


# ---------------------------
# Reward Model Head
# ---------------------------
class RewardHead(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h):
        return self.linear(h)


rm_head = RewardHead()


# ---------------------------
# Ranking loss (pairwise)
# ---------------------------
def ranking_loss(r_w, r_l):
    return -torch.log(torch.sigmoid(r_w - r_l))


# ---------------------------
# Dummy dataset (PoC)
# ---------------------------
prompt = "Explain RLHF"
chosen = "RLHF stands for Reinforcement Learning from Human Feedback..."
rejected = "I don't know."

h_chosen = get_embedding_with_dropout(prompt + chosen)
h_rejected = get_embedding_with_dropout(prompt + rejected)


# ---------------------------
# Compute RM loss
# ---------------------------
r_w = rm_head(h_chosen)
r_l = rm_head(h_rejected)
L_RM = ranking_loss(r_w, r_l)


# ---------------------------
# Compute contrastive loss
# ---------------------------
# positive: same sentence different dropout
h_pos1 = get_embedding_no_dropout(chosen)       
h_pos2 = get_embedding_with_dropout(chosen)     # dropout noisy representation


# negatives: some random texts
neg1 = get_embedding_no_dropout("other text")
neg2 = get_embedding_no_dropout("more unrelated text")
negatives = torch.stack([neg1, neg2])

L_contrast = contrastive_loss(h_pos1, h_pos2, negatives)


# ---------------------------
# Total Loss
# ---------------------------
beta = 1.0
L_total = L_RM + beta * L_contrast
print("L_RM:", L_RM.item())
print("L_contrast:", L_contrast.item())
print("Total Loss:", L_total.item())


# ---------------------------
# Backprop
# ---------------------------
optimizer = torch.optim.Adam(rm_head.parameters(), lr=1e-4)

RM_log = []
cont_log = []


for step in range(2000):   # 跑 200 steps 當作示範
    optimizer.zero_grad()

    # ---- 1. Get embeddings ----
    h_w_clean = get_embedding_no_dropout(chosen)      # winner clean
    h_w_drop  = get_embedding_with_dropout(chosen)    # winner dropout
    
    h_l_clean = get_embedding_no_dropout(rejected)    # loser clean
    h_l_drop  = get_embedding_with_dropout(rejected)  # loser dropout

    negatives = torch.stack([
        get_embedding_with_dropout("random 1"),
        get_embedding_with_dropout("random 2"),
        get_embedding_with_dropout("random 3"),
    ])

    # ---- 2. Reward outputs ----
    reward_w = rm_head(h_w_clean)
    reward_l = rm_head(h_l_clean)

    L_RM = ranking_loss(reward_w, reward_l)
    L_contrast = contrastive_loss(h_w_clean, h_w_drop, negatives)

    RM_log.append(L_RM.item())
    cont_log.append(L_contrast.item())

    L_total = L_RM + beta * L_contrast

    # ---- 3. Backprop ----
    L_total.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | L_RM: {L_RM.item():.4f} | L_cont: {L_contrast.item():.4f} | Total: {L_total.item():.4f}")

plt.plot(RM_log)
plt.plot(cont_log)
plt.show()

# print("\nBackprop success! Parameters updated.")
