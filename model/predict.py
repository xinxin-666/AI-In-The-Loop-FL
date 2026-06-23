import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载你训练好的模型
MODEL_PATH = "results/scam_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("Loading model from: results/scam_model")

# 输入消息
text = input("Enter message: ")

# 编码
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# 预测
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

# 分类结果
prediction = torch.argmax(outputs.logits, dim=1).item()

# 概率
probabilities = torch.softmax(outputs.logits, dim=1)
confidence = probabilities[0][prediction].item() * 100

print("raw logits:", outputs.logits)

probabilities = torch.softmax(outputs.logits, dim=1)
print("probabilities:", probabilities)

print("prediction:", prediction)

# 输出
if prediction == 1:
    print(f"SCAM ({confidence:.2f}%)")
else:
    print(f"NOT SCAM ({confidence:.2f}%)")