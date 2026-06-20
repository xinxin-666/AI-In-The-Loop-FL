import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# 读数据
df = pd.read_csv("data/datasets/train.csv")

texts = df["message"].tolist()
labels = df["label"].tolist()

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# 自定义数据集
class SpamDataset(Dataset):
    def __init__(self,texts,labels):
        self.texts=texts
        self.labels=labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):

        encoding=tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids":encoding["input_ids"].squeeze(),
            "attention_mask":encoding["attention_mask"].squeeze(),
            "labels":torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }

dataset=SpamDataset(texts,labels)

# 每次只喂8条
loader=DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

device="cuda" if torch.cuda.is_available() else "cpu"

model=AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(device)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

for epoch in range(3):

    model.train()

    total_loss=0

    for batch in loader:

        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["labels"].to(device)

        outputs=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss=outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(
        f"Epoch {epoch+1}: Loss={total_loss:.4f}"
    )

print("Training Finished")

model.save_pretrained(
    "results/scam_model"
)

tokenizer.save_pretrained(
    "results/scam_model"
)

print("Model Saved")