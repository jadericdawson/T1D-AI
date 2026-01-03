"""
ISFNet Model
Ported from ISFNet_model.py

LSTM-based model for predicting Insulin Sensitivity Factor (ISF).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISFNet(nn.Module):
    """
    Insulin Sensitivity Factor Predictor using LSTM.

    Architecture:
    - LSTM: n_feat -> hidden_size, num_layers with dropout
    - Linear: hidden_size -> 1
    - Softplus: Ensures output > 0 (ISF must be positive)

    Note: The trained model uses larger dimensions than the default.
    Check the saved model's state_dict for actual dimensions.
    """

    def __init__(
        self,
        n_feat: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_feat = n_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_feat)

        Returns:
            ISF predictions of shape (batch, 1), guaranteed > 0
        """
        # LSTM forward pass
        _, (h_n, _) = self.lstm(x)

        # Use hidden state from last layer
        logits = self.fc(h_n[-1])  # Shape: (batch, 1)

        # Softplus ensures output is always positive
        return F.softplus(logits)


# Model configuration (may need to be updated based on actual trained model)
ISF_MODEL_CONFIG = {
    "n_feat": 26,  # Should match the feature count used in training
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
}

# Note: The actual trained model dimensions should be verified
# by loading the state_dict and checking layer sizes
