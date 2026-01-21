"""
Temporal Fusion Transformer (TFT) for Blood Glucose Prediction

State-of-the-art architecture for multi-horizon time series forecasting
with built-in uncertainty quantification.

Key advantages over LSTM:
1. Attention-based - captures long-range temporal dependencies
2. Variable selection - learns which features are important
3. Quantile outputs - provides uncertainty bands (10th, 50th, 90th percentiles)
4. Interpretable - attention weights show what the model focuses on

Research: Edge-TFT achieves RMSE of 19.09 mg/dL @ 30min, 32.31 mg/dL @ 60min

This implementation can use pytorch-forecasting TFT or a lightweight custom version.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class TFTPrediction:
    """TFT prediction output with uncertainty bounds."""
    horizon_min: int           # Prediction horizon in minutes
    value: float               # Median prediction (50th percentile)
    lower: float               # Lower bound (10th percentile)
    upper: float               # Upper bound (90th percentile)
    confidence: float          # Confidence level (default 0.8 for 10th-90th)


@dataclass
class TFTOutput:
    """Complete TFT prediction output."""
    predictions: List[TFTPrediction]
    attention_weights: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - core building block of TFT.

    Provides non-linear processing with gating mechanism for
    controlling information flow.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size

        # Primary transformation
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Context transformation (optional)
        if context_size is not None:
            self.fc_context = nn.Linear(context_size, hidden_size, bias=False)

        # Second layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Gating layer (GLU)
        self.fc_gate = nn.Linear(hidden_size, output_size * 2)

        # Residual connection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None

        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Primary transformation
        hidden = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_size is not None:
            hidden = hidden + self.fc_context(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gated Linear Unit (GLU)
        gate_input = self.fc_gate(hidden)
        value, gate = gate_input.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)

        # Residual connection
        if self.skip is not None:
            residual = self.skip(x)
        else:
            residual = x

        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - learns feature importance.

    Outputs:
    - Selected features (weighted combination)
    - Feature importance weights (for interpretability)
    """

    def __init__(
        self,
        input_size: int,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()

        self.num_features = num_features

        # Transform each input feature
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
            for _ in range(num_features)
        ])

        # Softmax gate for variable selection
        self.softmax_gate = nn.Sequential(
            nn.Linear(num_features * hidden_size, num_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, num_features)
        batch_size, seq_len, _ = x.shape

        # Transform each feature
        transformed = []
        for i, grn in enumerate(self.feature_grns):
            feature = x[:, :, i:i+1]  # (batch, seq_len, 1)
            transformed.append(grn(feature))  # (batch, seq_len, hidden)

        # Stack and flatten for gate
        stacked = torch.stack(transformed, dim=-1)  # (batch, seq_len, hidden, num_features)
        flattened = stacked.permute(0, 1, 3, 2).reshape(batch_size * seq_len, -1)

        # Compute importance weights
        weights = self.softmax_gate(flattened)  # (batch*seq_len, num_features)
        weights = weights.view(batch_size, seq_len, self.num_features)

        # Weighted combination
        selected = (stacked * weights.unsqueeze(2)).sum(dim=-1)  # (batch, seq_len, hidden)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)

        return output, avg_attn


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for glucose prediction.

    Architecture:
    1. Variable Selection Network - learns feature importance
    2. LSTM Encoder - captures local temporal patterns
    3. Transformer Attention - captures long-range dependencies
    4. Quantile Output - provides uncertainty bounds

    Outputs predictions at multiple horizons (30, 45, 60 min)
    with quantile estimates (10th, 50th, 90th percentiles).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        encoder_length: int = 24,     # 120 min history (24 * 5 min)
        prediction_length: int = 12,  # 60 min prediction (12 * 5 min)
        quantiles: List[float] = [0.1, 0.5, 0.9],  # 10th, 50th, 90th
        horizons_minutes: List[int] = [30, 45, 60],  # Prediction horizons in minutes
        timestep_minutes: int = 5  # Time step size in minutes
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.horizons_minutes = horizons_minutes
        self.timestep_minutes = timestep_minutes

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_size=1,
            num_features=n_features,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # LSTM Encoder for local temporal patterns
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=encoder_length + prediction_length)

        # Transformer attention layers
        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.attention_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )

        # Output layers for each quantile
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in quantiles
        ])

        # Compute horizon indices from horizons_minutes
        # Each horizon in minutes / timestep_minutes gives the 1-indexed position
        # Subtract 1 for 0-indexed
        self.horizon_indices = [(h // timestep_minutes) - 1 for h in horizons_minutes]
        self.n_horizons = len(horizons_minutes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
            return_attention: Whether to return attention weights

        Returns:
            Dict containing:
            - 'predictions': (batch, n_horizons, n_quantiles) tensor
            - 'attention_weights': Optional attention weights
            - 'feature_importance': Optional feature importance
        """
        batch_size, seq_len, _ = x.shape

        # Variable Selection
        selected, feature_importance = self.vsn(x)

        # Add positional encoding
        selected = self.pos_encoding(selected)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_grn(attn_out)

        # Generate quantile outputs
        quantile_outputs = []
        for output_layer in self.output_layers:
            q_out = output_layer(attn_out)  # (batch, seq_len, 1)
            quantile_outputs.append(q_out)

        # Stack quantiles: (batch, seq_len, n_quantiles)
        all_quantiles = torch.cat(quantile_outputs, dim=-1)

        # Extract predictions at target horizons
        # For a 24-step input, the last step represents "now"
        # Predictions are at relative positions from end of encoder
        predictions = []
        for idx in self.horizon_indices:
            # Use the last encoder_length positions for prediction
            pred_idx = min(idx, seq_len - 1)
            horizon_pred = all_quantiles[:, -self.prediction_length + idx, :]
            predictions.append(horizon_pred)

        # Stack horizons: (batch, n_horizons, n_quantiles)
        predictions = torch.stack(predictions, dim=1)

        result = {'predictions': predictions}

        if return_attention:
            result['attention_weights'] = attn_weights.detach().cpu().numpy()
            result['feature_importance'] = feature_importance[:, -1, :].detach().cpu().numpy()

        return result

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> TFTOutput:
        """
        Generate predictions with uncertainty bounds.

        Args:
            x: Input tensor of shape (1, seq_len, n_features)

        Returns:
            TFTOutput with predictions at 30, 45, 60 minutes
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_attention=True)

        predictions = output['predictions'].squeeze(0).numpy()  # (n_horizons, n_quantiles)
        horizons = [30, 45, 60]

        results = []
        for i, horizon in enumerate(horizons):
            pred = TFTPrediction(
                horizon_min=horizon,
                value=float(predictions[i, 1]),   # 50th percentile (median)
                lower=float(predictions[i, 0]),   # 10th percentile
                upper=float(predictions[i, 2]),   # 90th percentile
                confidence=0.8  # 10th to 90th = 80% confidence interval
            )
            results.append(pred)

        return TFTOutput(
            predictions=results,
            attention_weights=output.get('attention_weights'),
            feature_importance=output.get('feature_importance')
        )


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.

    Pinball loss that penalizes predictions differently based on
    whether they over or under-predict.
    """

    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantile loss.

        Args:
            predictions: (batch, n_horizons, n_quantiles)
            targets: (batch, n_horizons, 1) or (batch, n_horizons)

        Returns:
            Scalar loss tensor
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        losses = []
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, :, i:i+1]
            error = targets - pred
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)

        return torch.cat(losses, dim=-1).mean()


# Model configuration
TFT_MODEL_CONFIG = {
    "n_features": 69,           # Extended feature set (trained model uses 69)
    "hidden_size": 64,
    "n_heads": 4,
    "n_lstm_layers": 2,
    "dropout": 0.1,
    "encoder_length": 24,       # 120 min history
    "prediction_length": 12,    # 60 min prediction
    "quantiles": [0.1, 0.5, 0.9],
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "horizons_minutes": [30, 45, 60],
}


def create_tft_model(n_features: int = 69) -> TemporalFusionTransformer:
    """Create a TFT model with default configuration."""
    return TemporalFusionTransformer(
        n_features=n_features,
        hidden_size=TFT_MODEL_CONFIG["hidden_size"],
        n_heads=TFT_MODEL_CONFIG["n_heads"],
        n_lstm_layers=TFT_MODEL_CONFIG["n_lstm_layers"],
        dropout=TFT_MODEL_CONFIG["dropout"],
        encoder_length=TFT_MODEL_CONFIG["encoder_length"],
        prediction_length=TFT_MODEL_CONFIG["prediction_length"],
        quantiles=TFT_MODEL_CONFIG["quantiles"]
    )
