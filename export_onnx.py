import torch
from dqn import DQN

model = DQN()
model.load_state_dict(torch.load("mancala_model.pth"))
model.eval()

torch.onnx.export(
    model,
    (torch.randn(1, 14),),  # dummy input
    "mancala_model.onnx",
    input_names=["state"],
    output_names=["q_values"],
    opset_version=18
)