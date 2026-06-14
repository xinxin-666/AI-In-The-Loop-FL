import flwr as fl
import numpy as np

class BertClient(fl.client.NumPyClient):

    def __init__(self):
        self.parameters = [
            np.random.randn(10, 10),
            np.random.randn(10)
        ]

    def get_parameters(self, config):
        print("Client1: 发送模型参数")
        return self.parameters

    def fit(self, parameters, config):

        print("Client1: 本地训练开始")

        self.parameters = [
            p + np.random.normal(0, 0.01, p.shape)
            for p in parameters
        ]

        print("Client1: 本地训练结束")

        return self.parameters, 24, {}

    def evaluate(self, parameters, config):

        print("Client1: 本地评估")

        return 0.1, 24, {"accuracy": 0.90}

client = BertClient()

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client
)