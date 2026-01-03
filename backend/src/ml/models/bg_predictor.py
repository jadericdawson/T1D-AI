"""
BG_PredictorNet Model
Ported from train_bg_predictor.py:322-334

LSTM-based model for predicting blood glucose at +5, +10, +15 minutes.
"""
import torch
import torch.nn as nn


class BG_PredictorNet(nn.Module):
    """
    Blood Glucose Predictor using LSTM.

    Architecture:
    - LSTM: n_features -> hidden_size, num_layers with dropout
    - Linear: hidden_size -> out_steps (3 for +5m, +10m, +15m)

    Default configuration matches trained model:
    - hidden_size=128
    - num_layers=3
    - dropout=0.2
    """

    def __init__(
        self,
        n_features: int,
        out_steps: int = 3,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout_prob: float = 0.2
    ):
        super().__init__()

        self.n_features = n_features
        self.out_steps = out_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, out_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features)

        Returns:
            Predictions of shape (batch, out_steps)
        """
        # LSTM forward pass
        lstm_out, (hidden_state, cell_state) = self.lstm(x)

        # Use the hidden state from the last layer
        last_layer_hidden = hidden_state[-1]  # Shape: (batch, hidden_size)

        # Project to output
        output = self.fc(last_layer_hidden)  # Shape: (batch, out_steps)

        return output


# Model configuration constants (matching trained model)
BG_MODEL_CONFIG = {
    "n_features": 26,
    "out_steps": 3,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout_prob": 0.2,
    "seq_length": 24,  # 120 min / 5 min
    "sampling_min": 5,
    "history_min": 120,
}

# Feature column names (must match training)
BG_FEATURE_COLUMNS = [
    "value", "trend",
    "carbs", "protein", "fat",
    "iob",
    "roll30_mean", "roll30_std", "roll90_mean", "roll90_std",
    "secs_since_start",
    "min5_sin", "min5_cos", "min15_sin", "min15_cos",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "mon_sin", "mon_cos", "doy_sin", "doy_cos",
    "poly0", "poly1", "poly2"
]
