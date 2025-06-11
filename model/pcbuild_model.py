"""Neural network model for PC component recommendation."""

import torch.nn as nn


class PCBuildModel(nn.Module):
    """Multitask neural network model for predicting PC components."""

    def __init__(self, encoders):
        """
        Initialize the PCBuildModel.

        Args:
            encoders (dict): Dictionary of fitted LabelEncoders for each component.
        """
        super().__init__()
        self.encoders = encoders
        self.input_size = 3  # price + game_score + work_score
        hidden_size = 128

        self.shared = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.output_heads = nn.ModuleDict({
            "CPU": nn.Linear(hidden_size, len(encoders["CPU"].classes_)),
            "Motherboard": nn.Linear(hidden_size, len(encoders["Motherboard"].classes_)),
            "Memory": nn.Linear(hidden_size, len(encoders["Memory"].classes_)),
            "Video Card": nn.Linear(hidden_size, len(encoders["Video Card"].classes_)),
            "Power Supply": nn.Linear(hidden_size, len(encoders["Power Supply"].classes_)),
        })

    def forward(self, inputs):
        """
        Perform forward pass through the network.

        Args:
            inputs (torch.Tensor): Tensor of shape (batch_size, 3) with normalized inputs.

        Returns:
            dict: Dictionary mapping component names to raw logits.
        """
        shared_features = self.shared(inputs)
        return {
            component: head(shared_features)
            for component, head in self.output_heads.items()
        }
