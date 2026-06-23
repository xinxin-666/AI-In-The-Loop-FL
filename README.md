# AI-In-The-Loop-FL

AI-In-The-Loop-FL is a federated learning based scam message detection system using BERT for text classification. The project combines privacy-preserving distributed learning and AI-based text analysis to identify fraudulent messages while keeping user data localized.

---

# Project Overview

This project simulates a federated learning environment for scam detection. Multiple clients train models locally on their own datasets and only exchange model parameters with the central server.

Main features:

* Federated Learning using Flower
* BERT-based text classification
* Distributed training with multiple clients
* Privacy-preserving architecture
* Scam message detection
* Non-IID data distribution simulation

---

# System Architecture

Client devices:

* Client1
* Client2

Central server:

* Flower Federated Server

Model:

* BERT (bert-base-uncased)

Training process:

1. Server initializes global model
2. Clients receive parameters
3. Clients train locally
4. Parameters sent back to server
5. Server aggregates using FedAvg
6. Repeat for multiple rounds

---

# Dataset

The project uses manually collected and constructed text samples.

Total dataset:

* Total samples: 48
* Scam messages: 24
* Normal messages: 24

Client distribution:

Client1:

* Total: 24
* Scam: 20
* Normal: 4

Client2:

* Total: 24
* Scam: 4
* Normal: 20

Dataset is intentionally distributed in a Non-IID manner to simulate realistic federated learning environments where users possess different local data distributions.

---

# Environment

Required packages:

```bash
pip install torch
pip install transformers
pip install flwr
pip install pandas
pip install opacus
```

Or:

```bash
pip install -r requirements.txt
```

---

# Project Structure

```text
AI-In-The-Loop-FL/
│
├── client1_bert.py
├── client2_bert.py
├── server.py
├── model/
│   └── predict.py
│
├── data/
│   ├── client1/
│   │    └── scam_messages.csv
│   │
│   └── client2/
│        └── scam_messages.csv
│
├── results/
│   └── scam_model/
│
└── README.md
```

---

# Training

Start server:

```bash
python server.py
```

Open separate terminals and start clients:

Client1:

```bash
python client1_bert.py
```

Client2:

```bash
python client2_bert.py
```

Training configuration:

* Federated rounds: 10
* Optimizer: AdamW
* Learning rate: 2e-5
* Model: BERT-base

---

# Model Performance

Training loss:

| Round | Loss       |
| ----- | ---------- |
| 1     | 0.2783     |
| 2     | 0.0780     |
| 3     | 0.0176     |
| 4     | 0.1526     |
| 5     | 0.000025   |
| 6     | 0.0000049  |
| 7     | 0.0000018  |
| 8     | 0.00000089 |
| 9     | 0.00000036 |
| 10    | 0.00000018 |

The model converged successfully after federated training.

---

# Prediction Examples

Normal messages:

Input:

```text
Hello, how are you?
```

Output:

```text
NOT SCAM
Confidence: 100%
```

Input:

```text
Let's meet tomorrow.
```

Output:

```text
NOT SCAM
Confidence: 100%
```

Scam messages:

Input:

```text
Your account has been locked.
```

Output:

```text
SCAM
Confidence: 100%
```

Input:

```text
Verify your password now.
```

Output:

```text
SCAM
Confidence: 100%
```

---

# Inference

Run prediction:

```bash
python model/predict.py
```

Example:

```text
Enter message:
Your account has been locked.
```

Output:

```text
SCAM (100%)
```

Average runtime:

* Approximately 5.32 seconds

Note:

Current runtime includes:

* Python startup
* Model loading
* Prediction

Actual inference time after model loading is significantly lower.

---

# Model Files

Trained model weights are stored locally:

```text
results/scam_model/
```

Large model files are not uploaded to GitHub because of repository file-size limitations.

Users can retrain locally to generate model weights.

---

# Future Improvements

* Increase dataset size
* Add more clients
* Introduce stronger differential privacy mechanisms
* Improve model generalization
* Deploy as Web service
* Real-time scam detection support

---

# Authors

AI-In-The-Loop-FL Team
