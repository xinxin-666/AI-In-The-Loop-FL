import torch
import flwr as fl
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "results/scam_model"
)

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_parameters = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is not None:

            parameters_aggregated, _ = aggregated_parameters

            ndarrays = fl.common.parameters_to_ndarrays(
                parameters_aggregated
            )

            params_dict = zip(
                model.state_dict().keys(),
                ndarrays
            )

            state_dict = {
                k: torch.tensor(v)
                for k, v in params_dict
            }

            model.load_state_dict(
                state_dict,
                strict=True
            )

            model.save_pretrained(
                "results/scam_model"
            )

            print(
                f"Round {server_round}: model saved"
            )

        return aggregated_parameters


strategy = SaveModelStrategy()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(
        num_rounds=10
    )
)