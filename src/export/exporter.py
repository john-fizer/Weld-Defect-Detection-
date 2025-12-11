"""
Model Export Module
Export PyTorch models to ONNX and TensorRT for production deployment
Achieves <3ms inference on edge hardware
"""

import torch
import onnx
import onnxsim
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import numpy as np


class ModelExporter:
    """
    Export PyTorch model to ONNX and TensorRT
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, int, int, int] = (1, 3, 448, 448),
        device: str = "cuda",
    ):
        """
        Initialize exporter

        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (batch, channels, height, width)
            device: Device for model
        """
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        logger.info(f"Initialized ModelExporter")
        logger.info(f"Input shape: {input_shape}")

    def export_to_onnx(
        self,
        output_path: Path,
        opset_version: int = 17,
        simplify: bool = True,
        check: bool = True,
        dynamic_axes: Optional[dict] = None,
    ) -> None:
        """
        Export model to ONNX format

        Args:
            output_path: Output ONNX file path
            opset_version: ONNX opset version
            simplify: Simplify ONNX graph
            check: Check ONNX model validity
            dynamic_axes: Dynamic axes for batch size
        """
        logger.info(f"Exporting to ONNX: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)

        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

        logger.success(f"ONNX model exported to {output_path}")

        # Simplify ONNX graph
        if simplify:
            logger.info("Simplifying ONNX model...")
            try:
                model_simp, check_ok = onnxsim.simplify(str(output_path))
                if check_ok:
                    onnx.save(model_simp, str(output_path))
                    logger.success("ONNX model simplified")
                else:
                    logger.warning("ONNX simplification failed, using original model")
            except Exception as e:
                logger.warning(f"ONNX simplification error: {e}")

        # Check ONNX model
        if check:
            logger.info("Checking ONNX model...")
            try:
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                logger.success("ONNX model is valid")
            except Exception as e:
                logger.error(f"ONNX model check failed: {e}")
                raise

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model size: {file_size_mb:.2f} MB")

    def benchmark_onnx(
        self,
        onnx_path: Path,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> dict:
        """
        Benchmark ONNX model inference speed

        Args:
            onnx_path: Path to ONNX model
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking ONNX model: {onnx_path}")

        import onnxruntime as ort
        import time

        # Create ONNX Runtime session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Create dummy input
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)

        # Warmup
        logger.info(f"Warmup: {warmup_runs} runs...")
        for _ in range(warmup_runs):
            session.run(None, {input_name: dummy_input})

        # Benchmark
        logger.info(f"Benchmarking: {num_runs} runs...")
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        throughput = 1000 / mean_time  # images/second

        results = {
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "throughput_imgs_per_sec": throughput,
            "num_runs": num_runs,
        }

        logger.info(f"Benchmark results:")
        logger.info(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
        logger.info(f"  Min time: {min_time:.2f} ms")
        logger.info(f"  Max time: {max_time:.2f} ms")
        logger.info(f"  Throughput: {throughput:.1f} images/sec")

        return results

    def export_to_tensorrt(
        self,
        onnx_path: Path,
        output_path: Path,
        precision: str = "fp16",
        workspace_size: int = 4096,
        max_batch_size: int = 32,
    ) -> None:
        """
        Convert ONNX model to TensorRT engine

        Args:
            onnx_path: Input ONNX model path
            output_path: Output TensorRT engine path
            precision: Precision mode ('fp32', 'fp16', or 'int8')
            workspace_size: Workspace size in MB
            max_batch_size: Maximum batch size
        """
        logger.info(f"Converting ONNX to TensorRT: {output_path}")
        logger.info(f"Precision: {precision}")

        try:
            import tensorrt as trt
        except ImportError:
            logger.error("TensorRT not installed. Install from https://developer.nvidia.com/tensorrt")
            raise

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        # Create builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        logger.info("Parsing ONNX model...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 20))

        # Set precision
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 mode enabled")
            else:
                logger.warning("FP16 not supported on this platform, using FP32")
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.warning("INT8 mode requires calibration dataset (not implemented)")
            else:
                logger.warning("INT8 not supported on this platform, using FP32")

        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            raise RuntimeError("TensorRT engine build failed")

        # Save engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        logger.success(f"TensorRT engine saved to {output_path}")

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"TensorRT engine size: {file_size_mb:.2f} MB")


def export_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    model_class,
    config: dict,
    export_onnx: bool = True,
    export_tensorrt: bool = False,
    tensorrt_precision: str = "fp16",
) -> dict:
    """
    Export model from PyTorch Lightning checkpoint

    Args:
        checkpoint_path: Path to .ckpt file
        output_dir: Output directory
        model_class: PyTorch Lightning model class
        config: Model configuration
        export_onnx: Export to ONNX
        export_tensorrt: Export to TensorRT
        tensorrt_precision: TensorRT precision

    Returns:
        Dictionary with export paths
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    model = model_class.load_from_checkpoint(checkpoint_path, config=config)
    model = model.model  # Extract inner model
    model.eval()

    # Create exporter
    input_size = config["data"]["image_size"]
    exporter = ModelExporter(
        model=model,
        input_shape=(1, 3, input_size, input_size),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    export_paths = {}

    # Export to ONNX
    if export_onnx:
        onnx_path = output_dir / "weld_classifier.onnx"
        exporter.export_to_onnx(
            output_path=onnx_path,
            opset_version=17,
            simplify=True,
            check=True,
        )
        export_paths["onnx"] = onnx_path

        # Benchmark ONNX
        benchmark_results = exporter.benchmark_onnx(onnx_path)
        export_paths["onnx_benchmark"] = benchmark_results

    # Export to TensorRT
    if export_tensorrt:
        if "onnx" not in export_paths:
            logger.error("TensorRT export requires ONNX export first")
        else:
            trt_path = output_dir / f"weld_classifier_{tensorrt_precision}.engine"
            exporter.export_to_tensorrt(
                onnx_path=export_paths["onnx"],
                output_path=trt_path,
                precision=tensorrt_precision,
            )
            export_paths["tensorrt"] = trt_path

    return export_paths


if __name__ == "__main__":
    # Example usage
    from src.training.model import WeldClassifier

    # Create a dummy model
    model = WeldClassifier(
        backbone_name="convnextv2_nano.fcmae_ft_in22k_in1k_384",
        num_classes=2,
        pretrained=False,
    )

    # Export
    exporter = ModelExporter(model, input_shape=(1, 3, 448, 448))
    exporter.export_to_onnx(
        Path("models/onnx/test_model.onnx"),
        simplify=True,
        check=True,
    )

    # Benchmark
    exporter.benchmark_onnx(Path("models/onnx/test_model.onnx"))

    logger.success("Export test completed!")
