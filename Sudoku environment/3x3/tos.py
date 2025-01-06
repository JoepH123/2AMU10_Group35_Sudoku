import torch
import pickle
import numpy as np
import onnx
import os

# Installeer evt. onnx (indien nog niet gebeurd): pip install onnx
# Installeer onnxruntime voor inference (in aparte script): pip install onnxruntime

# Voorbeeld van je CNN-model (vervang door je eigen definitie als nodig)
# of importeer vanuit je code, bijvoorbeeld:
# from dqn import CNNQNetwork
class CNNQNetwork(torch.nn.Module):
    """
    A Convolutional Neural Network for Q-value estimation (DQN).
    Expects a (3, 9, 9) input (three channels, 9x9 board).
    Outputs Q-values for 81 discrete actions (9x9 possible moves).
    """
    def __init__(self, input_shape=(9,9), num_actions=81):
        super(CNNQNetwork, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )
        # Flatten: 64 * 9 * 9 = 5184
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(5184, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_pkl_model(pkl_filename="dqn_model.pkl"):
    """
    Laadt een PyTorch-model uit een pickle-bestand met 'policy_state_dict'.

    """

    # Haal de directory van het huidige script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Combineer met de bestandsnaam
    pkl_path = os.path.join(script_dir, pkl_filename)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    model = CNNQNetwork(input_shape=(9,9), num_actions=81)
    model.load_state_dict(data["policy_state_dict"])  # Let op de dict-sleutelnaam
    model.eval()
    return model

def export_to_onnx(model, onnx_filename="dqn_model.onnx"):
    """
    Exporteer PyTorch-model naar ONNX. We gebruiken een dummy_input (batch_size=1, 3x9x9).
    """
    dummy_input = torch.randn(1, 3, 9, 9)  # batch=1, channels=3, 9x9 board
    model.eval()

    # Exporteer naar ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename,
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13  # kies een geschikte opset, 13 is gangbaar
    )
    print(f"Model is geëxporteerd naar ONNX-bestand: {onnx_filename}")

if __name__ == "__main__":
    # 1. Laad je .pkl-model
    model = load_pkl_model("team35_9x9_dqn_model.pkl")  # <-- pas evt. bestandsnaam aan

    # 2. Exporteer naar ONNX
    export_to_onnx(model, "team35_9x9_dqn_model.onnx")
