"""Post-hoc collapse audit for the ORIGINAL soft-alignment checkpoints.

This script performs no training and does not modify checkpoints.  It reproduces
the original attention formula exactly:

    softmax((student_segment @ teacher_frame) / tau, dim=teacher_time)

Place it beside mmfi.py, PNL_reweight_Segment_Align.py and ResNet18_LSTM.py.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from mmfi import make_dataloader, make_dataset
from PNL_reweight_Segment_Align import PointNetLSTM
from ResNet18_LSTM import GrayTeacherNet


def preprocess_mmwave(batch, device):
    x = batch.get("input_mmwave")
    if x is None:
        raise RuntimeError("Batch missing input_mmwave")
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.float32).permute(0, 1, 3, 2).contiguous()


def original_soft_attention(student_seg, teacher_seq, tau=0.08):
    """Exact attention formula used in the original KD_Align.py."""
    if student_seg.ndim != 3 or teacher_seq.ndim != 3:
        raise ValueError("Expected student_seg=(B,K,D), teacher_seq=(B,T,D)")
    if student_seg.shape[0] != teacher_seq.shape[0] or student_seg.shape[2] != teacher_seq.shape[2]:
        raise ValueError(
            f"Incompatible feature shapes: {tuple(student_seg.shape)} and "
            f"{tuple(teacher_seq.shape)}"
        )
    similarity = torch.einsum("bkd,btd->bkt", student_seg, teacher_seq) / tau
    return torch.softmax(similarity, dim=-1)


def per_sample_metrics(attn):
    """Return one value per sample for every metric."""
    _, num_segments, teacher_steps = attn.shape

    # Fraction of segment pairs whose argmax is exactly the same Teacher frame.
    peaks = attn.argmax(dim=-1)
    pair_mask = torch.triu(
        torch.ones(num_segments, num_segments, dtype=torch.bool, device=attn.device),
        diagonal=1,
    )
    peak_equal = peaks.unsqueeze(2).eq(peaks.unsqueeze(1))
    peak_duplicate_rate = peak_equal[:, pair_mask].float().mean(dim=1)

    # A robust order metric based on attention barycenters, not unstable argmaxes.
    time = torch.linspace(0.0, 1.0, teacher_steps, dtype=attn.dtype, device=attn.device)
    centers = torch.sum(attn * time.view(1, 1, -1), dim=-1)
    adjacent_gaps = centers[:, 1:] - centers[:, :-1]
    order_violation_rate = (adjacent_gaps <= 0.0).float().mean(dim=1)

    # Mean cosine similarity between all pairs of complete attention rows.
    unit_attn = F.normalize(attn, p=2, dim=-1, eps=1e-12)
    overlap_matrix = torch.bmm(unit_attn, unit_attn.transpose(1, 2))
    attention_overlap = overlap_matrix[:, pair_mask].mean(dim=1)

    # Auxiliary diagnostics: high entropy can make argmax duplication misleading.
    safe_attn = attn.clamp_min(1e-12)
    normalized_entropy = (
        -(safe_attn * safe_attn.log()).sum(dim=-1) / math.log(teacher_steps)
    ).mean(dim=1)
    mean_adjacent_gap = adjacent_gaps.mean(dim=1)
    max_attention = attn.max(dim=-1).values.mean(dim=1)

    return {
        "peak_duplicate_rate": peak_duplicate_rate,
        "order_violation_rate": order_violation_rate,
        "attention_overlap": attention_overlap,
        "normalized_entropy": normalized_entropy,
        "mean_adjacent_gap": mean_adjacent_gap,
        "mean_max_attention": max_attention,
        "centers": centers,
        "peaks": peaks,
    }


def load_state_dict_robust(model, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_over_samples": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def analyze_fold(config, data_root, teacher_root, student_root, fold, tau, device):
    config["kfold"]["current_fold"] = fold
    generator = torch.Generator().manual_seed(config.get("seed", 0) + fold)
    _, _, test_dataset = make_dataset(data_root, config)
    test_loader = make_dataloader(
        test_dataset,
        False,
        generator=generator,
        batch_size=config.get("analysis_batch_size", 16),
    )

    teacher_path = os.path.join(teacher_root, f"fold_{fold}", "best_res18_pretrain.pth")
    student_path = os.path.join(student_root, f"fold_{fold}", "best_student_softalign.pth")
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    if not os.path.isfile(student_path):
        raise FileNotFoundError(f"Student checkpoint not found: {student_path}")

    teacher = GrayTeacherNet(num_classes=27).to(device)
    student = PointNetLSTM(num_classes=27).to(device)
    load_state_dict_robust(teacher, teacher_path, device, strict=False)
    load_state_dict_robust(student, student_path, device, strict=True)
    teacher.eval()
    student.eval()

    scalar_rows = []
    sample_rows = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"Original alignment audit - fold {fold}"):
            mmwave = preprocess_mmwave(batch, device)
            rgb = batch["input_rgb"].unsqueeze(2).float().to(device)
            _, teacher_seq = teacher(rgb, return_features=True)
            _, student_seg = student(mmwave, return_features=True)
            attn = original_soft_attention(student_seg, teacher_seq, tau=tau)
            metrics = per_sample_metrics(attn)

            batch_size = attn.shape[0]
            for index in range(batch_size):
                scalar = {
                    key: float(metrics[key][index].cpu())
                    for key in (
                        "peak_duplicate_rate",
                        "order_violation_rate",
                        "attention_overlap",
                        "normalized_entropy",
                        "mean_adjacent_gap",
                        "mean_max_attention",
                    )
                }
                scalar_rows.append(scalar)
                sample_rows.append({
                    "action": batch["action"][index],
                    "subject": batch["subject"][index],
                    **scalar,
                    "attention_centers": metrics["centers"][index].cpu().tolist(),
                    "peak_frames_zero_based": metrics["peaks"][index].cpu().tolist(),
                })

    metric_names = scalar_rows[0].keys()
    summary = {
        "fold": fold,
        "num_test_samples": len(sample_rows),
        "teacher_checkpoint": teacher_path,
        "student_checkpoint": student_path,
        "tau": tau,
        "metrics": {
            name: summarize([row[name] for row in scalar_rows]) for name in metric_names
        },
        "samples": sample_rows,
    }
    del teacher, student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def analyze_all_folds(config_path, data_root, teacher_root, student_root, output_path, tau=0.08):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = config["kfold"]["num_folds"]
    print(f"Using device: {device}; auditing {num_folds} folds")

    folds = []
    for fold in range(num_folds):
        fold_result = analyze_fold(
            config=config,
            data_root=data_root,
            teacher_root=teacher_root,
            student_root=student_root,
            fold=fold,
            tau=tau,
            device=device,
        )
        folds.append(fold_result)
        short = {name: values["mean"] for name, values in fold_result["metrics"].items()}
        print(f"Fold {fold}: {json.dumps(short, ensure_ascii=False)}")

    metric_names = folds[0]["metrics"].keys()
    overall = {}
    for name in metric_names:
        per_fold_means = [fold["metrics"][name]["mean"] for fold in folds]
        all_sample_values = [
            sample[name] for fold in folds for sample in fold["samples"]
        ]
        overall[name] = {
            "mean_over_all_fold_sample_pairs": float(np.mean(all_sample_values)),
            "mean_of_fold_means": float(np.mean(per_fold_means)),
            "std_of_fold_means": float(np.std(per_fold_means, ddof=1)) if num_folds > 1 else 0.0,
        }

    report = {
        "method": "original raw-dot-product soft temporal alignment",
        "attention_formula": "softmax(einsum(student_seg, teacher_seq) / tau, dim=-1)",
        "num_folds": num_folds,
        "overall": overall,
        "folds": folds,
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n=== ORIGINAL SOFT-ALIGNMENT AUDIT ===")
    for name, values in overall.items():
        print(
            f"{name}: {values['mean_of_fold_means']:.6f} "
            f"+/- {values['std_of_fold_means']:.6f} (std across folds)"
        )
    print(f"Full report saved to: {output_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="data/MMFi")
    parser.add_argument(
        "--teacher-root", default="checkpoints/rgb_baseline"
    )
    parser.add_argument(
        "--student-root", default="checkpoints/mmwave_distilled"
    )
    parser.add_argument(
        "--output",
        default="checkpoints/mmwave_distilled/original_alignment_audit.json",
    )
    parser.add_argument("--tau", type=float, default=0.08)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_all_folds(
        config_path=args.config,
        data_root=args.data_root,
        teacher_root=args.teacher_root,
        student_root=args.student_root,
        output_path=args.output,
        tau=args.tau,
    )
