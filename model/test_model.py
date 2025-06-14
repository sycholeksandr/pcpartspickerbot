"""Test script for running PCBuildModel recommendation."""

import torch
import joblib

from model.pcbuild_model import PCBuildModel
from bot.recommender import recommend_parts
from config.settings import MODEL_PATH, ENCODERS_PATH
import logging

logging.basicConfig(
    level=logging.INFO,  # або DEBUG для детальнішого виводу
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def main():
    """Load model and encoders, then generate and print a PC build recommendation."""
    encoders = joblib.load(ENCODERS_PATH)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PCBuildModel(encoders=encoders)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    for i in range(10):
        print(f"Test {i+1}:")
        price = 500 + i * 200
        task = "work" if i % 2 == 0 else "games"
        build = recommend_parts(price, task)
        print("🔧 Рекомендована збірка:")
        for component, value in build.items():
            print(f"{component}: {value}")

if __name__ == "__main__":
    main()
