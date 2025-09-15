# Acceleration (Intel XPU/CPU)

Deep learning models can run accelerated on Intel GPUs as an optional enhancement.

- See `src/ml/xpu_optimizer.py` for device selection and optimizations
- CPU remains the default path; acceleration is opt-in

Checklist:
- Install required drivers and PyTorch XPU build
- Set device in config or environment
- Validate parity between CPU and XPU runs

