import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "results/scam_model"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# 检测函数
def detect_scam(text):

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
        return f"SCAM ({confidence:.2f}%)"
    else:
        return f"NOT SCAM ({confidence:.2f}%)"


demo = gr.Interface(
    fn=detect_scam,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter a message..."
    ),
    outputs="text",
    title="FedGuard Scam Detection System",
    description="Federated Learning + DistilBERT"
)

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False
)