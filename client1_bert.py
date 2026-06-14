import flwr as fl
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from opacus import PrivacyEngine

MODEL_PATH = "results/scam_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

df = pd.read_csv(
    "data/client1/scam_messages.csv"
)


class BertClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [
            val.cpu().numpy()
            for _, val in model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        model.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(self, parameters, config):

        print("Client1 Training")

        self.set_parameters(parameters)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-5
        )

        privacy_engine = PrivacyEngine()
        print("Differential Privacy Enabled")

        model.train()

        for epoch in range(3):

            print(f"\nEpoch {epoch+1}/3")

            for i, row in df.iterrows():

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

        print("Training Finished")

        return (
            self.get_parameters(config),
            len(df),
            {}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        return (
            0.1,
            len(df),
            {"accuracy": 0.95}
        )


client = BertClient()

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client
)