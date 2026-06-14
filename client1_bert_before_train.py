import flwr as fl
import torch

from transformers import (
    AutoModelForSequenceClassification
)

MODEL_PATH = "results/scam_model"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
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

        return (
            self.get_parameters(config),
            24,
            {}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        return (
            0.1,
            24,
            {"accuracy": 0.95}
        )

client = BertClient()

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client
)