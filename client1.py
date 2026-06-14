import flwr as fl
import numpy as np

class ScamClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [np.random.randn(5)]

    def fit(self, parameters, config):
        print("客户端1开始训练")
        return parameters, 1, {}

    def evaluate(self, parameters, config):
        print("客户端1开始评估")
        return 0.1, 1, {}

client = ScamClient()

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client
)