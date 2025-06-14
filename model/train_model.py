"""Train PCBuildModel on final_cleaned_builds.csv and save model and encoders."""

import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from .pcbuild_model import PCBuildModel  # або ./pcbuild_model, якщо запускаєш з локального каталогу
import logging
from config.settings import DATASET_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class BuildDataset(Dataset):
    """Custom Dataset for PC build features and labels."""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {
            key: torch.tensor(value[idx], dtype=torch.long)
            for key, value in self.targets.items()
        }

def main():
    """Main training loop for PCBuildModel."""
    df = pd.read_csv(DATASET_PATH)
    df["is_top_segment"] = (df["Total Price"] >= 4500).astype(float)
    categorical_columns = ["CPU", "Motherboard", "Memory", "Video Card", "Power Supply"]
    numeric_columns = ["Total Price", "Game Score", "Work Score", "is_top_segment"]

    encoders = {
        col: LabelEncoder().fit(df[col]) for col in categorical_columns
    }
    for col in categorical_columns:
        df[col] = encoders[col].transform(df[col])

    features = df[numeric_columns].values.astype(np.float32)
    targets = {col: df[col].values for col in categorical_columns}
    targets["IsTopSegment"] = df["is_top_segment"].values.astype(np.int64)

    dataset = BuildDataset(features, targets)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = PCBuildModel(encoders=encoders).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = {k: v.to(device) for k, v in batch_y.items()}

            predictions = model(batch_x)
            losses = [loss_fn(predictions[k], batch_y[k]) for k in predictions]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "model/pcbuild_model.pt")
    with open("model/encoders.pkl", "wb") as file:
        pickle.dump(encoders, file)

if __name__ == "__main__":
    main()
