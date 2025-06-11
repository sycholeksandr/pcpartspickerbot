"""Test script for running PCBuildModel recommendation."""

import torch
import joblib

from model.pcbuild_model import PCBuildModel
from bot.recommender import recommend_parts
from config.settings import MODEL_PATH, ENCODERS_PATH

def main():
    """Load model and encoders, then generate and print a PC build recommendation."""
    encoders = joblib.load(ENCODERS_PATH)


    # Example input
    # Adjust these values as needed for testing
    price = 1900
    task = "work"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PCBuildModel(encoders=encoders)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    build = recommend_parts(price, task)
    print("🔧 Рекомендована збірка:")
    for component, value in build.items():
        print(f"{component}: {value}")

if __name__ == "__main__":
    main()
