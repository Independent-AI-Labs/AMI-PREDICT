#!/usr/bin/env python
"""
Advanced model architectures for high-frequency scalping with online learning support.
Optimized for Intel Arc A770 XPU.
"""

import random
from collections import deque

import numpy as np
import torch
from torch import nn

# Check for XPU support
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


class TCN(nn.Module):
    """Temporal Convolutional Network for scalping with online learning support."""

    def __init__(self, input_size: int, output_size: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_levels = len(num_channels)
        self.tcn_blocks = nn.ModuleList()

        for i in range(self.num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            self.tcn_blocks.append(self._make_tcn_block(in_channels, out_channels, kernel_size, dilation_size, dropout))

        self.fc = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def _make_tcn_block(self, in_channels, out_channels, kernel_size, dilation, dropout):
        """Create a TCN block with dilated causal convolution."""
        padding = (kernel_size - 1) * dilation

        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            x = x[:, :, : -x.size(2) + residual.size(2)]  # Causal padding

            # Residual connection
            if x.size(1) == residual.size(1):
                x = x + residual
            else:
                x = x + nn.Conv1d(residual.size(1), x.size(1), 1).to(x.device)(residual)

        # Global average pooling
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return self.sigmoid(x)

    def online_update(self, x, y, optimizer, criterion):
        """Perform online learning update."""
        self.train()
        optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()


class TransformerScalper(nn.Module):
    """Transformer with sliding window for high-frequency scalping."""

    def __init__(self, input_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, window_size: int = 100, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.window_size = window_size

    def _generate_positional_encoding(self, seq_len, d_model):
        """Generate positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape

        # Sliding window
        if seq_len > self.window_size:
            x = x[:, -self.window_size :, :]
            seq_len = self.window_size

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        pe = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pe

        # Transformer encoding
        x = self.transformer(x)

        # Use last token for prediction
        x = x[:, -1, :]
        x = self.fc(x)
        return self.sigmoid(x)


class OnlineLSTM(nn.Module):
    """LSTM with experience replay buffer for online learning."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, buffer_size: int = 10000):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.hidden_state = None

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, features)
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        # Use last output for prediction
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return self.sigmoid(output), hidden

    def add_to_buffer(self, x, y):
        """Add experience to replay buffer."""
        self.buffer.append((x, y))

    def sample_from_buffer(self, batch_size: int):
        """Sample batch from experience replay buffer."""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        x_batch = torch.cat([x for x, _ in batch])
        y_batch = torch.cat([y for _, y in batch])
        return x_batch, y_batch

    def online_update_with_replay(self, x, y, optimizer, criterion, replay_ratio=0.5):
        """Update with mix of new data and replay buffer."""
        self.train()

        # Add to buffer
        self.add_to_buffer(x, y)

        # Sample from buffer
        batch_size = x.size(0)
        replay_size = int(batch_size * replay_ratio)

        if len(self.buffer) > replay_size:
            replay_data = self.sample_from_buffer(replay_size)
            if replay_data:
                replay_x, replay_y = replay_data
                x = torch.cat([x, replay_x])
                y = torch.cat([y, replay_y])

        # Update
        optimizer.zero_grad()
        output, self.hidden_state = self(x, self.hidden_state)

        # Detach hidden state to prevent backprop through time
        if self.hidden_state is not None:
            self.hidden_state = tuple(h.detach() for h in self.hidden_state)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        return loss.item()


class EnsembleMetaLearner(nn.Module):
    """Ensemble that combines predictions from multiple models with dynamic weighting."""

    def __init__(self, models: dict[str, nn.Module], meta_features: int = 10):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.num_models = len(models)

        # Meta-learner for dynamic weight adjustment
        self.meta_learner = nn.Sequential(
            nn.Linear(self.num_models + meta_features, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, self.num_models), nn.Softmax(dim=1)
        )

        # Performance tracking
        self.model_performance = {name: deque(maxlen=100) for name in models.keys()}

    def forward(self, x, meta_features=None):
        """Forward pass with dynamic weighting."""
        predictions = []

        # Get predictions from all models
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                if isinstance(model, OnlineLSTM):
                    pred, _ = model(x)
                else:
                    pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=1).squeeze(-1)

        # Calculate dynamic weights
        if meta_features is not None:
            # Combine predictions with meta features
            meta_input = torch.cat([predictions, meta_features], dim=1)
            weights = self.meta_learner(meta_input)
        else:
            # Use uniform weights
            weights = torch.ones(x.size(0), self.num_models).to(x.device) / self.num_models

        # Weighted average
        ensemble_pred = (predictions * weights).sum(dim=1, keepdim=True)

        return ensemble_pred, weights

    def update_performance(self, model_name: str, accuracy: float):
        """Update model performance tracking."""
        self.model_performance[model_name].append(accuracy)

    def get_model_weights(self):
        """Get current average performance of each model."""
        weights = {}
        for name, perf in self.model_performance.items():
            if len(perf) > 0:
                weights[name] = np.mean(perf)
            else:
                weights[name] = 1.0 / self.num_models

        # Normalize
        total = sum(weights.values())
        for name in weights:
            weights[name] /= total

        return weights


class ScalpingModelFactory:
    """Factory for creating and managing scalping models."""

    @staticmethod
    def create_tcn(input_size: int, **kwargs) -> TCN:
        """Create TCN model."""
        return TCN(
            input_size=input_size,
            output_size=1,
            num_channels=kwargs.get("num_channels", [64, 128, 256]),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.2),
        ).to(DEVICE)

    @staticmethod
    def create_transformer(input_size: int, **kwargs) -> TransformerScalper:
        """Create Transformer model."""
        return TransformerScalper(
            input_size=input_size,
            d_model=kwargs.get("d_model", 256),
            nhead=kwargs.get("nhead", 8),
            num_layers=kwargs.get("num_layers", 4),
            window_size=kwargs.get("window_size", 100),
            dropout=kwargs.get("dropout", 0.1),
        ).to(DEVICE)

    @staticmethod
    def create_lstm(input_size: int, **kwargs) -> OnlineLSTM:
        """Create LSTM model."""
        return OnlineLSTM(
            input_size=input_size,
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
            buffer_size=kwargs.get("buffer_size", 10000),
        ).to(DEVICE)

    @staticmethod
    def create_ensemble(input_size: int, **kwargs) -> EnsembleMetaLearner:
        """Create ensemble model."""
        models = {
            "tcn": ScalpingModelFactory.create_tcn(input_size, **kwargs),
            "transformer": ScalpingModelFactory.create_transformer(input_size, **kwargs),
            "lstm": ScalpingModelFactory.create_lstm(input_size, **kwargs),
        }

        return EnsembleMetaLearner(models=models, meta_features=kwargs.get("meta_features", 10)).to(DEVICE)


def test_models():
    """Test all model architectures."""
    print("Testing scalping models on", DEVICE)

    # Create dummy data
    batch_size = 32
    seq_len = 100
    input_size = 50

    x = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
    y = torch.randint(0, 2, (batch_size, 1)).float().to(DEVICE)

    # Test each model
    factory = ScalpingModelFactory()

    print("\n1. Testing TCN...")
    tcn = factory.create_tcn(input_size)
    output = tcn(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in tcn.parameters()):,}")

    print("\n2. Testing Transformer...")
    transformer = factory.create_transformer(input_size)
    output = transformer(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    print("\n3. Testing LSTM...")
    lstm = factory.create_lstm(input_size)
    output, _ = lstm(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in lstm.parameters()):,}")

    print("\n4. Testing Ensemble...")
    ensemble = factory.create_ensemble(input_size)
    output, weights = ensemble(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in ensemble.parameters()):,}")

    print("\n✅ All models working on", DEVICE)


if __name__ == "__main__":
    test_models()
