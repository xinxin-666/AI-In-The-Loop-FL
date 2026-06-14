import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# 读取数据
df = pd.read_csv("data/scam_messages.csv")

texts = df["text"].tolist()
labels = torch.tensor(df["label"].tolist())

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# 模型
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# 训练5轮
for epoch in range(5):

    outputs = model(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=labels
    )

    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch+1} Loss = {loss.item():.4f}"
    )

print("\nTraining Finished")
model.save_pretrained("results/scam_model")
tokenizer.save_pretrained("results/scam_model")
print("Model Saved")