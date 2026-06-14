import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

MODEL_PATH = "results/scam_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

df = pd.read_csv(
    "data/test.csv"
)
texts = df["text"].tolist()
labels = df["label"].tolist()

predictions = []

for text in texts:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    predictions.append(pred)

accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions
)

recall = recall_score(
    labels,
    predictions
)

f1 = f1_score(
    labels,
    predictions
)

print("\n===== Evaluation Result =====")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)