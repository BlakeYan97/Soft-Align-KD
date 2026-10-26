"""Profile the Teacher, radar Student, and Soft-Align KD complexity.

This script does not modify or retrain any model. By default it uses the
architecture definitions in ``PNL_reweight_Segment_Align.py`` and
``ResNet18_LSTM.py`` with random weights, which is sufficient for parameter,
MAC/FLOP, memory, and latency measurements.

The deployed radar model excludes ``align_proj`` because this projection is
used only to construct the distillation loss. Removing it does not alter the
classification logits. Consequently, an independently trained Student and a
Soft-Align KD Student have exactly the same deployed architecture and cost.

Recommended server commands
---------------------------
Install the optional MAC profiler once::

    pip install thop

GPU profiling (recommended first run)::

    python profile_complexity.py --devices cuda

GPU and CPU latency profiling::

    python profile_complexity.py --devices cuda cpu --cpu-runs 20

The default inputs reproduce the paper protocol:
Student (1, 297, 5, 112), Teacher (1, 297, 1, 17, 2), and K=4.

Outputs are written to ``complexity_results/``:
``complexity_report.json``, ``complexity_table.csv``, and
``complexity_table.tex``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from PNL_reweight_Segment_Align import (
    PointNetLSTM, segment_max_pooling, resolve_point_mask,
)
from ResNet18_LSTM import GrayTeacherNet


class RadarStudentDeployment(nn.Module):
    """Inference-only Student with the distillation projection removed.

    The original ``PointNetLSTM.forward`` computes ``align_proj`` even when
    ``return_features=False``. This wrapper follows the same classification
    path but intentionally omits that training-only projection.
    """

    def __init__(self, training_model: PointNetLSTM, frame_chunk_size: int = 0):
        super().__init__()
        self.pointnet = training_model.pointnet
        self.lstm = training_model.lstm
        self.post_reweight = training_model.post_reweight
        self.classifier = training_model.classifier
        self.num_segments = training_model.num_segments
        self.frame_chunk_size = frame_chunk_size

    def forward(self, x: torch.Tensor, point_mask=None) -> torch.Tensor:
        batch_size, time_steps, channels, points = x.shape
        point_mask = resolve_point_mask(x, point_mask).reshape(-1, points)

        # PointNet processes frames independently. In eval mode, folding the
        # time axis into the batch axis is mathematically equivalent to the
        # original Python loop, while exposing parallelism to CPU/GPU kernels.
        # Optional chunking limits peak memory on smaller deployment devices.
        flat_frames = x.reshape(batch_size * time_steps, channels, points)
        if self.frame_chunk_size > 0:
            encoded_chunks = [
                self.pointnet(
                    flat_frames[start:start + self.frame_chunk_size],
                    point_mask=point_mask[start:start + self.frame_chunk_size],
                )
                for start in range(0, flat_frames.size(0), self.frame_chunk_size)
            ]
            encoded_frames = torch.cat(encoded_chunks, dim=0)
        else:
            encoded_frames = self.pointnet(flat_frames, point_mask=point_mask)
        frame_features = encoded_frames.reshape(batch_size, time_steps, -1)
        temporal_features, _ = self.lstm(frame_features)
        segment_features = segment_max_pooling(
            temporal_features, self.num_segments
        )
        segment_features = self.post_reweight(segment_features)
        sequence_feature, _ = segment_features.max(dim=1)
        return self.classifier(sequence_feature)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_if_given(
    model: nn.Module, checkpoint_path: Optional[str], model_name: str
) -> None:
    """Load common checkpoint layouts; weights are optional for profiling."""
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[Warning] {model_name} checkpoint loaded non-strictly. "
            f"Missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )


def make_student_training(args: argparse.Namespace) -> PointNetLSTM:
    model = PointNetLSTM(
        num_classes=args.num_classes,
        lstm_hidden=args.student_hidden,
        lstm_layers=1,
        bidirectional=True,
        num_segments=args.num_segments,
        align_dim=args.align_dim,
    )
    load_checkpoint_if_given(model, args.student_checkpoint, "Student")
    return model


def make_student_deployment(args: argparse.Namespace) -> RadarStudentDeployment:
    return RadarStudentDeployment(
        make_student_training(args), frame_chunk_size=args.frame_chunk_size
    )


def make_teacher(args: argparse.Namespace) -> GrayTeacherNet:
    # pretrained=False avoids downloading ImageNet weights. Weight values do
    # not affect parameter count, operations, or inference latency.
    model = GrayTeacherNet(
        num_classes=args.num_classes,
        lstm_hidden=args.teacher_hidden,
        lstm_layers=2,
        dropout=0.3,
        bidirectional=True,
        pretrained=False,
    )
    load_checkpoint_if_given(model, args.teacher_checkpoint, "Teacher")
    return model


def make_student_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randn(
        args.batch_size,
        args.sequence_length,
        5,
        args.points_per_frame,
        dtype=torch.float32,
        device=device,
    )


def make_teacher_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randn(
        args.batch_size,
        args.sequence_length,
        1,
        args.keypoint_count,
        args.keypoint_dimensions,
        dtype=torch.float32,
        device=device,
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def state_dict_size_mib(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024.0 ** 2)


def parameter_size_mib(model: nn.Module) -> float:
    size_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return size_bytes / (1024.0 ** 2)


def try_profile_macs(
    model: nn.Module, inputs: Tuple[torch.Tensor, ...]
) -> Tuple[Optional[float], Optional[str]]:
    try:
        from thop import profile  # type: ignore
    except ImportError:
        return None, "THOP is not installed; run: pip install thop"

    try:
        model.eval()
        with torch.inference_mode():
            macs, _ = profile(model, inputs=inputs, verbose=False)
        return float(macs), None
    except Exception as error:  # retain other results if THOP cannot trace an op
        return None, f"THOP profiling failed: {type(error).__name__}: {error}"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    model.eval()
    timings_ms: List[float] = []

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(*inputs)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
        else:
            baseline_memory = None

        for _ in range(measured_runs):
            synchronize(device)
            start = time.perf_counter()
            model(*inputs)
            synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

        if device.type == "cuda":
            peak_total = torch.cuda.max_memory_allocated(device)
            peak_increment = max(0, peak_total - int(baseline_memory or 0))
        else:
            peak_total = None
            peak_increment = None

    sorted_times = sorted(timings_ms)
    p95_index = max(0, min(len(sorted_times) - 1, int(0.95 * len(sorted_times)) - 1))
    return {
        "runs": measured_runs,
        "warmup_runs": warmup_runs,
        "mean_ms": statistics.mean(timings_ms),
        "std_ms": statistics.pstdev(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": sorted_times[p95_index],
        "peak_total_memory_mib": (
            peak_total / (1024.0 ** 2) if peak_total is not None else None
        ),
        "peak_forward_increment_mib": (
            peak_increment / (1024.0 ** 2)
            if peak_increment is not None else None
        ),
    }


def device_metadata(device: torch.device) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    else:
        metadata.update(
            {
                "processor": platform.processor(),
                "torch_num_threads": torch.get_num_threads(),
            }
        )
    return metadata


def clean_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def base_model_record(
    display_name: str,
    model: nn.Module,
    deployment_input: str,
    stage: str,
) -> Dict[str, Any]:
    parameters, trainable = count_parameters(model)
    return {
        "model": display_name,
        "stage": stage,
        "deployment_input": deployment_input,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_memory_fp32_mib": parameter_size_mib(model),
        "serialized_state_dict_mib": state_dict_size_mib(model),
        "macs": None,
        "gmacs": None,
        # Convention used here: one MAC equals two floating-point operations.
        "gflops_2_per_mac": None,
        "latency": {},
        "notes": [],
    }


def profile_architectures(args: argparse.Namespace) -> Dict[str, Any]:
    # Static model records are computed on CPU once.
    teacher = make_teacher(args).eval()
    full_student = make_student_training(args).eval()
    deployed_student = make_student_deployment(args).eval()

    records: Dict[str, Dict[str, Any]] = {
        "teacher": base_model_record(
            "Teacher (training only)", teacher, "Not deployed", "training"
        ),
        "student_training": base_model_record(
            "Student with alignment projection",
            full_student,
            "Radar",
            "KD training",
        ),
        "student_deployment": base_model_record(
            "Radar Student", deployed_student, "Radar only", "inference"
        ),
    }

    alignment_parameters = (
        records["student_training"]["parameters"]
        - records["student_deployment"]["parameters"]
    )
    records["student_training"]["notes"].append(
        f"Includes {alignment_parameters:,} training-only align_proj parameters."
    )
    records["student_deployment"]["notes"].append(
        "The independently trained Student and Soft-Align KD Student use this "
        "same inference architecture and therefore have identical inference cost."
    )
    records["student_deployment"]["notes"].append(
        "Frame-wise PointNet inference is vectorized over the folded batch-time "
        f"axis (frame_chunk_size={args.frame_chunk_size}; 0 means all frames)."
    )

    del teacher, full_student, deployed_student
    clean_memory(torch.device("cpu"))

    # MAC profiling uses the first requested available device. It can be
    # skipped independently from latency benchmarking.
    if not args.skip_macs:
        requested_flop_device = args.macs_device
        if requested_flop_device == "auto":
            requested_flop_device = "cuda" if torch.cuda.is_available() else "cpu"
        flop_device = torch.device(requested_flop_device)
        if flop_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--macs-device cuda requested but CUDA is unavailable")

        specs = [
            ("teacher", make_teacher, make_teacher_input),
            ("student_training", make_student_training, make_student_input),
            ("student_deployment", make_student_deployment, make_student_input),
        ]
        for key, model_factory, input_factory in specs:
            print(f"[MACs] Profiling {records[key]['model']} on {flop_device} ...")
            model = model_factory(args).to(flop_device).eval()
            inputs = (input_factory(args, flop_device),)
            macs, error = try_profile_macs(model, inputs)
            if macs is not None:
                records[key]["macs"] = macs
                records[key]["gmacs"] = macs / 1e9
                records[key]["gflops_2_per_mac"] = 2.0 * macs / 1e9
            if error:
                records[key]["notes"].append(error)
                print(f"[Warning] {error}")
            del model, inputs
            clean_memory(flop_device)

    # The two batched matrix multiplications dominate the explicitly coded
    # Soft-Align loss: student-to-teacher similarity and teacher aggregation.
    alignment_macs = (
        2
        * args.batch_size
        * args.num_segments
        * args.sequence_length
        * args.align_dim
    )

    # Latency is measured for deployed inference models. The teacher is also
    # measured as a training-only reference; the KD loss is not part of
    # deployment and is therefore represented analytically above.
    for device_name in args.devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA requested but unavailable; skipping CUDA latency.")
            continue
        device = torch.device(device_name)
        if device.type == "cpu" and args.cpu_threads > 0:
            torch.set_num_threads(args.cpu_threads)
        print(f"[Latency] Device: {device_metadata(device)}")

        latency_specs = [
            (
                "teacher",
                make_teacher,
                make_teacher_input,
                args.gpu_runs if device.type == "cuda" else args.cpu_runs,
            ),
            (
                "student_deployment",
                make_student_deployment,
                make_student_input,
                args.gpu_runs if device.type == "cuda" else args.cpu_runs,
            ),
        ]
        for key, model_factory, input_factory, measured_runs in latency_specs:
            print(
                f"[Latency] Benchmarking {records[key]['model']} on {device} "
                f"for {measured_runs} runs ..."
            )
            model = model_factory(args).to(device).eval()
            inputs = (input_factory(args, device),)
            try:
                latency = benchmark_latency(
                    model,
                    inputs,
                    device,
                    args.warmup_runs,
                    measured_runs,
                )
                latency["hardware"] = device_metadata(device)
                records[key]["latency"][str(device)] = latency
            except RuntimeError as error:
                records[key]["notes"].append(
                    f"Latency failed on {device}: {type(error).__name__}: {error}"
                )
                print(f"[Warning] Latency failed: {error}")
            del model, inputs
            clean_memory(device)

    # Add a separate publication-facing row for the distilled student. It is
    # deliberately a copy of the deployment Student record.
    distilled_record = json.loads(json.dumps(records["student_deployment"]))
    distilled_record["model"] = "Soft-Align KD Student"
    distilled_record["notes"].append(
        "Distillation changes learned weights, not the deployed architecture."
    )
    records["soft_align_kd_student"] = distilled_record

    teacher_params = records["teacher"]["parameters"]
    full_student_params = records["student_training"]["parameters"]
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "input_protocol": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "student_shape": [
                args.batch_size,
                args.sequence_length,
                5,
                args.points_per_frame,
            ],
            "teacher_shape": [
                args.batch_size,
                args.sequence_length,
                1,
                args.keypoint_count,
                args.keypoint_dimensions,
            ],
            "num_segments": args.num_segments,
            "align_dim": args.align_dim,
        },
        "models": records,
        "kd_training_only": {
            "loaded_teacher_plus_student_parameters": (
                teacher_params + full_student_params
            ),
            "trainable_student_parameters": full_student_params,
            "teacher_parameters_frozen": teacher_params,
            "soft_alignment_matrix_macs": alignment_macs,
            "soft_alignment_matrix_gmacs": alignment_macs / 1e9,
            "description": (
                "Approximate explicit Soft-Align matrix cost for the similarity "
                "and weighted aggregation operations. Softmax, normalization, "
                "MSE, and backward operations are not included."
            ),
        },
        "reporting_convention": (
            "THOP reports MACs. GFLOPs are additionally shown using the common "
            "convention 1 MAC = 2 FLOPs. State-dict size includes PyTorch "
            "serialization and non-trainable buffers."
        ),
    }
    return report


def format_number(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def latency_mean(record: Dict[str, Any], device: str) -> Optional[float]:
    item = record.get("latency", {}).get(device)
    return None if item is None else float(item["mean_ms"])


def publication_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = report["models"]
    keys = ["teacher", "student_deployment", "soft_align_kd_student"]
    rows: List[Dict[str, Any]] = []
    for key in keys:
        record = models[key]
        rows.append(
            {
                "Model": record["model"],
                "Parameters_M": record["parameters"] / 1e6,
                "GMACs": record["gmacs"],
                "GFLOPs_2_per_MAC": record["gflops_2_per_mac"],
                "FP32_Parameter_Size_MiB": record["parameter_memory_fp32_mib"],
                "Serialized_State_Dict_MiB": record["serialized_state_dict_mib"],
                "CPU_Latency_ms": latency_mean(record, "cpu"),
                "CUDA_Latency_ms": latency_mean(record, "cuda"),
                "Inference_Modality": record["deployment_input"],
            }
        )
    return rows


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Deployment complexity. Latency is measured with batch size one.}",
        r"\label{tab:deployment_complexity}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
        r"Model & Params (M) & GMACs & Size (MiB) & Latency (ms) & Input \\",
        r"\hline",
    ]
    for row in rows:
        latency = row["CUDA_Latency_ms"]
        if latency is None:
            latency = row["CPU_Latency_ms"]
        model_name = str(row["Model"]).replace("Soft-Align", r"Soft-Align")
        lines.append(
            f"{model_name} & "
            f"{format_number(row['Parameters_M'])} & "
            f"{format_number(row['GMACs'])} & "
            f"{format_number(row['FP32_Parameter_Size_MiB'], 2)} & "
            f"{format_number(latency)} & "
            f"{row['Inference_Modality']} " + r"\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== DEPLOYMENT COMPLEXITY SUMMARY ===")
    for row in publication_rows(report):
        print(
            f"{row['Model']}: "
            f"Params={row['Parameters_M']:.3f} M, "
            f"GMACs={format_number(row['GMACs'])}, "
            f"FP32 size={row['FP32_Parameter_Size_MiB']:.2f} MiB, "
            f"CPU={format_number(row['CPU_Latency_ms'])} ms, "
            f"CUDA={format_number(row['CUDA_Latency_ms'])} ms, "
            f"Input={row['Inference_Modality']}"
        )
    kd = report["kd_training_only"]
    print("\n=== KD TRAINING-ONLY OVERHEAD ===")
    print(
        "Teacher + full Student parameters loaded: "
        f"{kd['loaded_teacher_plus_student_parameters'] / 1e6:.3f} M"
    )
    print(
        "Explicit Soft-Align similarity/aggregation: "
        f"{kd['soft_alignment_matrix_gmacs']:.6f} GMACs per batch"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Teacher, Student, and Soft-Align KD complexity."
    )
    parser.add_argument("--output-dir", type=str, default="complexity_results")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--macs-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-macs", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--gpu-runs", type=int, default=50)
    parser.add_argument("--cpu-runs", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=297)
    parser.add_argument("--points-per-frame", type=int, default=112)
    parser.add_argument("--keypoint-count", type=int, default=17)
    parser.add_argument("--keypoint-dimensions", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=27)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument("--student-hidden", type=int, default=256)
    parser.add_argument("--teacher-hidden", type=int, default=512)
    parser.add_argument("--align-dim", type=int, default=1024)
    parser.add_argument(
        "--frame-chunk-size",
        type=int,
        default=0,
        help=(
            "Number of folded frames processed per PointNet call during deployment "
            "profiling; 0 processes all B*T frames together."
        ),
    )

    parser.add_argument("--student-checkpoint", type=str, default=None)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.devices is None:
        args.devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    for name in (
        "warmup_runs",
        "gpu_runs",
        "cpu_runs",
        "batch_size",
        "sequence_length",
        "points_per_frame",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = profile_architectures(args)
    rows = publication_rows(report)

    json_path = output_dir / "complexity_report.json"
    csv_path = output_dir / "complexity_table.csv"
    latex_path = output_dir / "complexity_table.tex"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_latex(rows, latex_path)
    print_summary(report)

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {latex_path}")


if __name__ == "__main__":
    main()
