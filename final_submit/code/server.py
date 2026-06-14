import flwr as fl
from flwr.server.strategy import FedAvg

strategy = FedAvg()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(
        num_rounds=3
    )
)