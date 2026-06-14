import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "results/scam_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

st.title("FedGuard")
st.subheader("Federated Learning Scam Detection System")

text = st.text_area("Enter Message")

if st.button("Detect"):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence = probs[0][prediction].item() * 100

    if prediction == 1:
        st.error(f"SCAM ({confidence:.2f}%)")
    else:
        st.success(f"NOT SCAM ({confidence:.2f}%)")