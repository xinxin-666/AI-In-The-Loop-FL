import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "results/scam_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

df = pd.read_csv(
    "data/client1/scam_messages.csv"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

model.train()

for i in range(len(df)):

    text = str(df.iloc[i]["text"])
    label = int(df.iloc[i]["label"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    outputs = model(
        **inputs,
        labels=torch.tensor([label])
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    print(
        f"Sample {i+1}/{len(df)} Loss={loss.item():.4f}"
    )

print("Training Finished")