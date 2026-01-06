import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import matplotlib.pyplot as plt
import os

device = ( "mps" if torch.backends.mps.is_available() else "cpu" )
print(f"Using device: {device}")

csv_filename = "pku_saferlhf_flip_harmfuleness_gap.csv"
if not os.path.exists(csv_filename):
    raise FileNotFoundError(csv_filename)

df = pd.read_csv(csv_filename)
df = df.fillna("").astype(str)




tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
print(model)

class RewardHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    
    def forward(self, x):
        return self.linear(x)
    

rm_head = RewardHead().to(device)
optimizer = torch.optim.Adam(rm_head.parameters(), lr=1e-4)

def get_embedding(texts, train=False):
    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0]
    

def ranking_loss(r_w, r_l):
    return -torch.log(torch.sigmoid(r_w - r_l)).mean()

def contrastive_loss(h1, h2, tau=0.05):
    sim = F.cosine_similarity(h1, h2)
    return -torch.log(torch.exp(sim / tau)).mean()


BATCH_SIZE = 16
EPOCHS = 5
beta = 1.0

RM_log, cont_log = [], []

log_file = open("clean_training_log.csv", "w", encoding="utf-8")
log_file.write("epoch,rm_loss,cont_loss\n")

step = 0

def train_reward_model(df, use_contrastive, epochs=3):
    for epoch in range(epochs):
        for i in range(0, len(df), BATCH_SIZE):
