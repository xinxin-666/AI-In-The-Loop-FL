import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

df = pd.read_csv(
    "data/client1/scam_messages.csv"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

model.train()

for epoch in range(3):

    print(f"\nEpoch {epoch+1}/3")

    for i,row in df.iterrows():

        text = row["text"]
        label = int(row["label"])

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Sample {i+1}/{len(df)} Loss={loss.item():.4f}"
        )

print("Single Training Finished")

model.save_pretrained(
    "results/single_model"
)

tokenizer.save_pretrained(
    "results/single_model"
)