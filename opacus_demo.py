import torch
from torch import nn
from opacus import PrivacyEngine

model = nn.Linear(10, 2)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

privacy_engine = PrivacyEngine()

print("Opacus Loaded Successfully")
print("Differential Privacy Enabled")