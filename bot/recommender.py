"""Recommender module using PCBuildModel for hardware prediction."""

import torch
import pickle
from model.pcbuild_model import PCBuildModel
from config.settings import MODEL_PATH, ENCODERS_PATH, DATASET_PATH
import logging
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

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

# Constants for scoring and price limits
df = pd.read_csv(DATASET_PATH)
MAX_GAME_SCORE = df["Game Score"].max() if "Game Score" in df.columns else 196.0
MAX_WORK_SCORE = df["Work Score"].max() if "Work Score" in df.columns else 203.0
MAX_PRICE = df["Total Price"].max() if "Total Price" in df.columns else 10575.64

def recommend_parts(price: float, task: str) -> dict:
    """
    Recommend PC parts based on budget and task type.

    Args:
        price (float): Budget in dollars.
        task (str): Task type, either "games" or "work".

    Returns:
        dict: Recommended components with readable labels.
    """
    [price, game_score, work_score, is_top_segment] = (
        prepare_scores_for_model_based_on_task(price, task).values()
    )
    input_tensor = torch.tensor(
        [[price, game_score, work_score, is_top_segment]], dtype=torch.float32
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
        logger.info(f"Readable model's recommendations: {readable}")
    return readable

def prepare_scores_for_model_based_on_task(price: float, task: str) -> dict:
        """
        Prepare the input for the model based on price and task type.

        Args:
            price (float): Budget in dollars.
            task (str): Task type, either "games" or "work".

        Returns:
            dict: Input tensor for the model.
        """
        is_top_segment = 1 if price >= 5000 else 0
        if price < 300.0 or price > MAX_PRICE:
            raise ValueError("Неможливо підібрати збірку з такою ціною.")
        if price < 4500.0:
            score_factor = min((price / 4500.0) ** 0.7, 0.9)
        else:
            score_factor = 1.0
        logger.info(f"Score factor based on price {price}: {score_factor:.2f}")
        if task == "games":
            game_score = MAX_GAME_SCORE * score_factor
            work_score = MAX_WORK_SCORE * score_factor * 0.9 
        elif task == "work":
            work_score = MAX_WORK_SCORE * score_factor
            game_score = MAX_GAME_SCORE * score_factor * 0.9
        else:
            # Змішане навантаження (наприклад, 50/50)
            game_score = MAX_GAME_SCORE * score_factor
            work_score = MAX_WORK_SCORE * score_factor
        return {
            "price": price,
            "game_score": game_score,
            "work_score": work_score,
            "is_top_segment": is_top_segment
        }