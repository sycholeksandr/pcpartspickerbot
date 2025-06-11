"""Recommender module using PCBuildModel for hardware prediction."""

import torch
import pickle
from model.pcbuild_model import PCBuildModel
from config.settings import MODEL_PATH, ENCODERS_PATH

# Load encoders
with open(ENCODERS_PATH, "rb") as file:
    ENCODERS = pickle.load(file)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
MODEL = PCBuildModel(encoders=ENCODERS)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()


def recommend_parts(price: float, task: str) -> dict:
    """
    Recommend PC parts based on budget and task type.

    Args:
        price (float): Budget in dollars.
        task (str): Task type, either "games" or "work".

    Returns:
        dict: Recommended components with readable labels.
    """
    game_score = 80.0 if task == "games" else 40.0
    work_score = 80.0 if task == "work" else 40.0

    input_tensor = torch.tensor(
        [[price, game_score, work_score]], dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(input_tensor)
        predictions = {
            key: torch.argmax(value, dim=1).item()
            for key, value in outputs.items()
        }
        readable = {
            key: ENCODERS[key].inverse_transform([predictions[key]])[0]
            for key in predictions
        }

    return readable

