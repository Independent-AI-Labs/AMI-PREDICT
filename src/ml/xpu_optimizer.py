#!/usr/bin/env python
"""
XPU optimization for scalping models.
Includes ONNX conversion, quantization, and Intel XPU kernel optimization.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.onnx
from torch import nn

# Intel extensions if available
try:
    import intel_extension_for_pytorch as ipex

    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# ONNX Runtime with XPU support
try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")


class XPUOptimizer:
    """Optimize models for Intel Arc A770 XPU."""

    def __init__(self, model_dir: str = "optimized_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def optimize_for_xpu(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """
        Optimize model for XPU execution.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor for tracing

        Returns:
            Optimized model
        """
        model.eval()
        model = model.to(DEVICE)
        sample_input = sample_input.to(DEVICE)

        if HAS_IPEX and DEVICE.type == "xpu":
            logger.info("Applying Intel Extension for PyTorch optimizations...")

            # Apply IPEX optimizations
            model = ipex.optimize(
                model,
                dtype=torch.float32,
                level="O1",  # O1 for mixed precision
                conv_bn_folding=True,
                linear_bn_folding=True,
                weights_prepack=True,
                replace_dropout_with_identity=True,
            )

            # JIT compile with XPU backend
            with torch.no_grad():
                model = torch.jit.trace(model, sample_input)
                model = torch.jit.optimize_for_inference(model)

            logger.info("XPU optimization complete")
        else:
            logger.info("IPEX not available, using standard optimizations")

            # Standard PyTorch optimizations
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)

        return model

    def export_to_onnx(self, model: nn.Module, sample_input: torch.Tensor, model_name: str) -> str:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            model_name: Name for the exported model

        Returns:
            Path to ONNX model
        """
        if not HAS_ONNX:
            logger.warning("ONNX Runtime not available")
            return None

        model.eval()
        onnx_path = self.model_dir / f"{model_name}.onnx"

        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        logger.info(f"Exported model to {onnx_path}")

        # Optimize ONNX model
        self._optimize_onnx(onnx_path)

        return str(onnx_path)

    def _optimize_onnx(self, onnx_path: Path):
        """Optimize ONNX model."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            # Load and optimize
            model = onnx.load(onnx_path)
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type="bert",  # General optimization
                num_heads=8,
                hidden_size=256,
                optimization_options={
                    "enable_gelu": True,
                    "enable_layer_norm": True,
                    "enable_attention": True,
                    "use_multi_head_attention": True,
                    "enable_skip_layer_norm": True,
                    "enable_embed_layer_norm": True,
                    "enable_bias_skip_layer_norm": True,
                    "enable_bias_gelu": True,
                    "enable_gelu_approximation": False,
                },
            )

            # Save optimized model
            optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
            optimized_model.save_model_to_file(str(optimized_path))

            logger.info(f"Optimized ONNX model saved to {optimized_path}")

        except Exception as e:
            logger.warning(f"Could not optimize ONNX model: {e}")

    def quantize_model(self, model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """
        Quantize model to INT8 for faster inference.

        Args:
            model: PyTorch model
            calibration_data: Data for calibration

        Returns:
            Quantized model
        """
        model.eval()

        if DEVICE.type == "xpu" and HAS_IPEX:
            logger.info("Applying INT8 quantization for XPU...")

            # Prepare for quantization
            from intel_extension_for_pytorch.quantization import convert, prepare

            qconfig = ipex.quantization.default_static_qconfig

            # Prepare model
            prepared_model = prepare(model, qconfig, example_inputs=calibration_data, inplace=False)

            # Calibrate
            with torch.no_grad():
                for i in range(min(100, len(calibration_data))):
                    prepared_model(calibration_data[i : i + 1])

            # Convert to quantized model
            quantized_model = convert(prepared_model)

            logger.info("Quantization complete")

        else:
            logger.info("Using PyTorch dynamic quantization...")

            # Dynamic quantization for CPU
            quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)

        return quantized_model

    def benchmark_inference(self, model: nn.Module, input_shape: tuple[int, ...], num_iterations: int = 1000) -> dict:
        """
        Benchmark model inference performance.

        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            num_iterations: Number of iterations for benchmarking

        Returns:
            Benchmark results
        """
        model.eval()

        # Create random input
        dummy_input = torch.randn(*input_shape).to(DEVICE)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

        # Benchmark
        if DEVICE.type == "xpu":
            torch.xpu.synchronize()

        start_time = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)

        if DEVICE.type == "xpu":
            torch.xpu.synchronize()

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = num_iterations / total_time

        results = {"device": str(DEVICE), "total_time": total_time, "avg_latency_ms": avg_latency, "throughput": throughput, "num_iterations": num_iterations}

        return results

    def create_inference_session(self, onnx_path: str) -> Optional["ort.InferenceSession"]:
        """
        Create ONNX Runtime inference session.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Inference session or None
        """
        if not HAS_ONNX:
            logger.warning("ONNX Runtime not available")
            return None

        # Set providers based on available hardware
        providers = []

        if DEVICE.type == "xpu":
            # Try OpenVINO for Intel GPU
            providers.append("OpenVINOExecutionProvider")

        # Fallback providers
        providers.extend(["CPUExecutionProvider"])

        try:
            session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"Created inference session with providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"Failed to create inference session: {e}")
            return None

    def benchmark_onnx_inference(self, session: "ort.InferenceSession", input_shape: tuple[int, ...], num_iterations: int = 1000) -> dict:
        """
        Benchmark ONNX Runtime inference.

        Args:
            session: ONNX Runtime session
            input_shape: Input shape
            num_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        if session is None:
            return {"error": "No inference session"}

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})

        # Benchmark
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            _ = session.run(None, {input_name: dummy_input})

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = num_iterations / total_time

        results = {
            "runtime": "ONNX Runtime",
            "providers": session.get_providers(),
            "total_time": total_time,
            "avg_latency_ms": avg_latency,
            "throughput": throughput,
            "num_iterations": num_iterations,
        }

        return results

    def compare_optimizations(self, model: nn.Module, input_shape: tuple[int, ...]) -> pd.DataFrame:
        """
        Compare different optimization techniques.

        Args:
            model: Original model
            input_shape: Input shape for testing

        Returns:
            DataFrame with comparison results
        """
        results = []
        sample_input = torch.randn(*input_shape).to(DEVICE)

        # 1. Original model
        logger.info("Benchmarking original model...")
        original_results = self.benchmark_inference(model, input_shape)
        original_results["optimization"] = "Original"
        results.append(original_results)

        # 2. XPU optimized
        logger.info("Benchmarking XPU optimized model...")
        xpu_model = self.optimize_for_xpu(model.clone(), sample_input)
        xpu_results = self.benchmark_inference(xpu_model, input_shape)
        xpu_results["optimization"] = "XPU Optimized"
        results.append(xpu_results)

        # 3. Quantized model
        if DEVICE.type == "cpu" or HAS_IPEX:
            logger.info("Benchmarking quantized model...")
            calibration_data = torch.randn(100, *input_shape[1:]).to(DEVICE)
            quantized_model = self.quantize_model(model.clone(), calibration_data)
            quant_results = self.benchmark_inference(quantized_model, input_shape)
            quant_results["optimization"] = "Quantized (INT8)"
            results.append(quant_results)

        # 4. ONNX Runtime
        if HAS_ONNX:
            logger.info("Benchmarking ONNX Runtime...")
            onnx_path = self.export_to_onnx(model, sample_input, "test_model")
            session = self.create_inference_session(onnx_path)
            if session:
                onnx_results = self.benchmark_onnx_inference(session, input_shape)
                onnx_results["optimization"] = "ONNX Runtime"
                results.append(onnx_results)

        # Create comparison DataFrame
        import pandas as pd

        df = pd.DataFrame(results)

        # Calculate speedup
        if len(df) > 0:
            baseline_latency = df.iloc[0]["avg_latency_ms"]
            df["speedup"] = baseline_latency / df["avg_latency_ms"]

        return df


def test_xpu_optimization():
    """Test XPU optimization on a sample model."""
    logger.info("Testing XPU Optimization")
    logger.info("=" * 60)

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_size=50, hidden_size=128):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return self.sigmoid(output)

    # Initialize
    model = SimpleModel().to(DEVICE)
    optimizer = XPUOptimizer()

    # Test input shape: (batch_size, seq_len, features)
    input_shape = (32, 100, 50)

    # Run comparison
    logger.info(f"\nComparing optimizations for input shape: {input_shape}")
    comparison_df = optimizer.compare_optimizations(model, input_shape)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPARISON RESULTS")
    logger.info("=" * 60)
    print(comparison_df.to_string())

    # Summary
    if len(comparison_df) > 1:
        best_idx = comparison_df["avg_latency_ms"].idxmin()
        best = comparison_df.loc[best_idx]

        logger.info("\n" + "=" * 60)
        logger.info("BEST CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Optimization: {best['optimization']}")
        logger.info(f"Average Latency: {best['avg_latency_ms']:.2f} ms")
        logger.info(f"Throughput: {best['throughput']:.0f} inferences/second")
        logger.info(f"Speedup: {best['speedup']:.2f}x")

    logger.info("\n✅ XPU optimization testing complete!")


if __name__ == "__main__":
    test_xpu_optimization()
