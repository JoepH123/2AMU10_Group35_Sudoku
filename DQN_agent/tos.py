import torch
import pickle
from DQN import CNNQNetwork  # Importeer jouw netwerkklasse

def load_pkl_model(pkl_path="dqn_model.pkl"):
    """
    Laadt een PyTorch-model uit een pickle-bestand.
    We gaan ervan uit dat het pickle-bestand een dict bevat
    met ten minste "policy_state_dict" voor de gewichten.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Instantieer jouw netwerk en laad de gewichten
    model = CNNQNetwork(input_shape=(9,9), num_actions=81)
    model.load_state_dict(data["policy_state_dict"])  # Let op de sleutelnaam!
    model.eval()
    return model

def convert_to_torchscript(model, ts_path="cnnqnetwork_scripted.pt"):
    """
    Converteert een PyTorch-model naar TorchScript via 'trace'.
    We maken een voorbeeldinput (dummy) van de juiste vorm: (1,3,9,9).
    """
    dummy_input = torch.randn(1, 3, 9, 9)  # batch_size=1, channels=3, board=9x9
    model.eval()

    # 'trace' het model met de dummy-invoer
    scripted_model = torch.jit.trace(model, dummy_input)
    
    # Sla het TorchScript-model op als .pt-bestand
    scripted_model.save(ts_path)
    print(f"TorchScript-model opgeslagen in: {ts_path}")

if __name__ == "__main__":
    # 1. Laad je pkl-model
    model = load_pkl_model("team35_9x9_dqn_model.pkl")

    # 2. Converteer naar TorchScript (cnnqnetwork_scripted.pt)
    convert_to_torchscript(model, "cnnqnetwork_scripted.pt")


