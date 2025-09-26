"""
Advanced Deep Learning Models for Crypto Trading
Supports both CPU and GPU training with automatic device selection
"""
import time
import warnings
from typing import Any, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class CryptoDataset(Dataset):
    """Custom dataset for crypto price data"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (self.X[idx : idx + self.sequence_length], self.y[idx + self.sequence_length])


class TransformerModel(nn.Module):
    """
    Advanced Transformer architecture for time series prediction
    Uses self-attention to capture long-range dependencies
    """

    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.decoder(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class LSTMAttentionModel(nn.Module):
    """
    LSTM with Attention mechanism for better feature extraction
    Combines temporal patterns with attention weights
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMAttentionModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Attention mechanism
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)

        return self.decoder(attended)


class WaveNet(nn.Module):
    """
    WaveNet-inspired architecture for capturing multi-scale temporal patterns
    Uses dilated convolutions for large receptive field
    """

    def __init__(self, input_dim: int, residual_channels: int = 64, skip_channels: int = 256, num_blocks: int = 4, layers_per_block: int = 10):
        super(WaveNet, self).__init__()

        self.input_conv = nn.Conv1d(input_dim, residual_channels, 1)

        self.residual_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for block in range(num_blocks):
            for layer in range(layers_per_block):
                dilation = 2**layer
                self.residual_blocks.append(ResidualBlock(residual_channels, skip_channels, dilation))

        self.output_network = nn.Sequential(nn.ReLU(), nn.Conv1d(skip_channels, skip_channels, 1), nn.ReLU(), nn.Conv1d(skip_channels, 1, 1))

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.input_conv(x)

        skip_sum = 0
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_sum += skip

        output = self.output_network(skip_sum)
        return output.squeeze(-1).squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block for WaveNet"""

    def __init__(self, residual_channels: int, skip_channels: int, dilation: int):
        super(ResidualBlock, self).__init__()

        self.dilated_conv = nn.Conv1d(residual_channels, residual_channels * 2, kernel_size=2, dilation=dilation, padding=dilation)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)

    def forward(self, x):
        dilated = self.dilated_conv(x)

        # Gated activation
        filter = torch.sigmoid(dilated[:, : dilated.size(1) // 2, :])
        gate = torch.tanh(dilated[:, dilated.size(1) // 2 :, :])
        activated = filter * gate

        skip = self.skip_conv(activated[:, :, :-1])
        residual = self.residual_conv(activated[:, :, :-1])

        return (x[:, :, :-1] + residual), skip


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    Efficient alternative to RNNs with parallelizable training
    """

    def __init__(self, input_dim: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super(TCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2**i

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.decoder = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, time)
        y = self.network(x)
        y = y[:, :, -1]  # Take last timestep
        return self.decoder(y)


class TemporalBlock(nn.Module):
    """Temporal block for TCN"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dropout1(self.conv1(x)[:, :, : -self.conv1.padding[0]]))
        out = self.relu(self.dropout2(self.conv2(out)[:, :, : -self.conv2.padding[0]]))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class DeepLearningTrainer:
    """Trainer for deep learning models with GPU support"""

    def __init__(self, model_type: str = "transformer", use_gpu: bool = True):
        self.model_type = model_type
        self.device = DEVICE if use_gpu else torch.device("cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = []

    def create_model(self, input_dim: int) -> nn.Module:
        """Create model based on type"""

        if self.model_type == "transformer":
            model = TransformerModel(input_dim, d_model=256, nhead=8, num_layers=4)
        elif self.model_type == "lstm_attention":
            model = LSTMAttentionModel(input_dim, hidden_dim=256, num_layers=3)
        elif self.model_type == "wavenet":
            model = WaveNet(input_dim, residual_channels=64, num_blocks=3)
        elif self.model_type == "tcn":
            model = TCN(input_dim, num_channels=[64, 128, 256, 128, 64])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model.to(self.device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        sequence_length: int = 60,
    ) -> dict[str, Any]:
        """Train the model"""

        print(f"\nTraining {self.model_type} on {self.device}")
        start_time = time.time()

        # Normalize data
        X_train = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        if X_val is not None:
            X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        # Create datasets
        train_dataset = CryptoDataset(X_train, y_train, sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = CryptoDataset(X_val, y_val, sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        input_dim = X_train.shape[-1]
        self.model = self.create_model(input_dim)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            if X_val is not None:
                self.model.eval()
                val_loss = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_x).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"Early stopping at epoch {epoch}")
                        break

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}")

        # Load best model
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Move to CPU for inference if needed
        if self.device.type == "cuda":
            self.model = self.model.cpu()
            self.device = torch.device("cpu")

        return {
            "training_time": training_time,
            "final_train_loss": avg_train_loss,
            "best_val_loss": best_val_loss if X_val is not None else None,
            "model_type": self.model_type,
            "device_used": str(DEVICE),
        }

    def predict(self, X: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        """Make predictions"""

        if self.model is None:
            raise ValueError("Model not trained yet")

        # Normalize
        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # Create dataset
        dataset = CryptoDataset(X, np.zeros(len(X)), sequence_length)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save(self, path: str):
        """Save model and scaler"""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_type": self.model_type,
                "scaler": self.scaler,
                "input_dim": self.model.input_projection.in_features if hasattr(self.model, "input_projection") else None,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and scaler"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model_type = checkpoint["model_type"]
        self.scaler = checkpoint["scaler"]

        # Recreate model
        if checkpoint["input_dim"]:
            self.model = self.create_model(checkpoint["input_dim"])
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

        print(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    X_train = np.random.randn(1000, 30)  # 1000 samples, 30 features
    y_train = np.random.randn(1000)

    # Test each model type
    for model_type in ["transformer", "lstm_attention", "tcn"]:
        print(f"\n{'='*60}")
        print(f"Testing {model_type}")
        print("=" * 60)

        trainer = DeepLearningTrainer(model_type=model_type)
        results = trainer.train(X_train, y_train, epochs=10, batch_size=32)
        print(f"Results: {results}")

        # Make predictions
        predictions = trainer.predict(X_train[:100])
        print(f"Predictions shape: {predictions.shape}")
