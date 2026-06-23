import flwr as fl
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "results/scam_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)



class BertClient(fl.client.NumPyClient):
    def __init__(self):

        self.model = model
        self.tokenizer = tokenizer

    def get_parameters(self, config):
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(self, parameters, config):

      print("Client1 Training")

      self.set_parameters(parameters)

      self.model.train()

      optimizer = torch.optim.AdamW(
          self.model.parameters(),
          lr=2e-5
      )

      df = pd.read_csv("data/client1/scam_messages.csv")
      df = df.sample(frac=1).reset_index(drop=True)

      for epoch in range(1):

          print(f"\nEpoch {epoch+1}/1")

          for i, row in df.iterrows():

              text = row["text"]
              label = torch.tensor([int(row["label"])])

              inputs = self.tokenizer(
                  text,
                  return_tensors="pt",
                  truncation=True,
                  padding=True
              )

              outputs = self.model(
                  **inputs,
                  labels=label
              )

              loss = outputs.loss

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              print(f"Sample {i+1}/{len(df)} Loss={loss.item():.4f}")

      print("Training Finished")

      return (
          self.get_parameters(config),
          len(df),
          {}
      )

    def evaluate(self, parameters, config):

      self.set_parameters(parameters)

      df = pd.read_csv("data/client1/scam_messages.csv")

      self.model.eval()

      total_loss = 0

      with torch.no_grad():

          for _, row in df.iterrows():

              text = row["text"]
              label = int(row["label"])

              inputs = self.tokenizer(
                  text,
                  return_tensors="pt",
                  truncation=True,
                  padding=True
              )

              outputs = self.model(
                  **inputs,
                  labels=torch.tensor([label])
              )

              total_loss += outputs.loss.item()

      avg_loss = total_loss / len(df)

      return (
          avg_loss,
          len(df),
          {}
      )
if __name__ == "__main__":


    client = BertClient()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )