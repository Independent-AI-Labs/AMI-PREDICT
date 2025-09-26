#!/usr/bin/env python
"""
Advanced model architectures for improved trading performance.
Includes attention mechanisms, WaveNet, and hybrid models.
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear transformations and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)
        return output


class AttentionTCN(nn.Module):
    """TCN with self-attention layers for improved pattern recognition."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 4, num_heads: int = 8):
        super().__init__()

        # TCN layers with increasing dilation
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            in_channels = input_size if i == 0 else hidden_size
            self.tcn_layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=dilation, dilation=dilation))

        # Attention layers
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size * 4, hidden_size))

        # Output layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        # TCN feature extraction
        for i, layer in enumerate(self.tcn_layers):
            residual = x
            x = F.relu(layer(x))
            x = self.dropout(x)

            # Residual connection if dimensions match
            if residual.size(1) == x.size(1):
                x = x + residual

        # Transpose for attention
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)

        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Global pooling and output
        x = x.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.global_pool(x).squeeze(-1)

        return torch.sigmoid(self.output(x))


class WaveNetModel(nn.Module):
    """WaveNet-style architecture for time series prediction."""

    def __init__(self, input_size: int, residual_channels: int = 32, skip_channels: int = 256, num_blocks: int = 3, layers_per_block: int = 10):
        super().__init__()

        self.input_conv = nn.Conv1d(input_size, residual_channels, kernel_size=1)

        self.residual_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for block in range(num_blocks):
            for layer in range(layers_per_block):
                dilation = 2**layer

                # Dilated convolution
                self.residual_blocks.append(nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=2, dilation=dilation, padding=dilation))

                # Skip connection
                self.skip_connections.append(nn.Conv1d(residual_channels, skip_channels, kernel_size=1))

        # Output network
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        x = self.input_conv(x)
        skip_sum = 0

        for i, (residual_layer, skip_layer) in enumerate(zip(self.residual_blocks, self.skip_connections, strict=False)):
            residual = x

            # Gated activation
            conv_out = residual_layer(x)
            filter_gate, gate = conv_out.chunk(2, dim=1)
            x = torch.tanh(filter_gate) * torch.sigmoid(gate)

            # Skip connection
            skip_sum = skip_sum + skip_layer(x)

            # Residual connection (handle dimension mismatch)
            if x.size(-1) == residual.size(-1):
                x = x + residual
            else:
                # Trim or pad to match dimensions
                min_len = min(x.size(-1), residual.size(-1))
                x = x[..., :min_len] + residual[..., :min_len]

        # Output
        x = F.relu(skip_sum)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)

        # Global average pooling
        x = x.mean(dim=-1)

        return torch.sigmoid(x)


class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM with attention for feature extraction and sequence modeling."""

    def __init__(self, input_size: int, cnn_channels: int = 64, lstm_hidden: int = 128, num_layers: int = 2):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels * 2, cnn_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.ReLU(),
        )

        # LSTM sequence modeling
        self.lstm = nn.LSTM(cnn_channels * 4, lstm_hidden, num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=True)

        # Attention mechanism
        self.attention = nn.Sequential(nn.Linear(lstm_hidden * 2, lstm_hidden), nn.Tanh(), nn.Linear(lstm_hidden, 1))

        # Output layer
        self.output = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        cnn_out = self.cnn(x)

        # Prepare for LSTM
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len//4, features)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)

        # Attention weights
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)

        # Output
        return torch.sigmoid(self.output(context))


class TransformerTrader(nn.Module):
    """Pure transformer architecture for trading."""

    def __init__(self, input_size: int, d_model: int = 256, num_heads: int = 8, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.register_buffer("positional_encoding", self._generate_positional_encoding(1000, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.output = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model // 2, 1))

    def _generate_positional_encoding(self, max_len: int, d_model: int):
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        # x: (batch, seq_len, features)
        seq_len = x.size(1)

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)

        # Use last position for prediction
        x = x[:, -1, :]

        return torch.sigmoid(self.output(x))


class GRUAttention(nn.Module):
    """GRU with Bahdanau attention for sequence modeling."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=True)

        # Bahdanau attention
        self.attention = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

        self.output = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, _ = x.size()

        # GRU encoding
        gru_out, hidden = self.gru(x)

        # Get last hidden state
        hidden = hidden.view(batch_size, -1)  # Concatenate bidirectional

        # Attention mechanism
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([gru_out, hidden_expanded], dim=-1)

        attn_weights = self.attention(combined)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Context vector
        context = torch.sum(gru_out * attn_weights, dim=1)

        return torch.sigmoid(self.output(context))


def test_models():
    """Test all advanced models."""
    import time

    # Check device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Testing on Intel XPU: {torch.xpu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Testing on CPU")

    # Test parameters
    batch_size = 32
    seq_len = 100
    input_size = 10

    # Create dummy data
    x = torch.randn(batch_size, seq_len, input_size).to(device)

    models = [
        ("AttentionTCN", AttentionTCN(input_size)),
        ("WaveNetModel", WaveNetModel(input_size)),
        ("HybridCNNLSTM", HybridCNNLSTM(input_size)),
        ("TransformerTrader", TransformerTrader(input_size)),
        ("GRUAttention", GRUAttention(input_size)),
    ]

    print("\n" + "=" * 60)
    print("ADVANCED MODELS TEST")
    print("=" * 60)

    for name, model in models:
        model = model.to(device)
        model.eval()

        # Test forward pass
        start = time.time()
        with torch.no_grad():
            output = model(x)

        if device.type == "xpu":
            torch.xpu.synchronize()

        elapsed = time.time() - start

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params:,}")
        print(f"  Inference time: {elapsed*1000:.2f}ms")
        print(f"  Throughput: {batch_size/elapsed:.0f} samples/sec")

    print("\n✅ All models working!")


if __name__ == "__main__":
    test_models()
