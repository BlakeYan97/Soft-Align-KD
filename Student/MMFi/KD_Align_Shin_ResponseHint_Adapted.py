"""Common-backbone response + hint distillation adapted from Shin et al.

Shin et al., "Graph Neural Network Compression via Cross-Modal Distillation
for On-Device mmWave Human Activity Recognition", IEEE Access, 2025,
DOI: 10.1109/ACCESS.2025.3635398.

This is a comparison of the paper's transfer objectives under this project's
PointNet-BiLSTM student and ResNet18-BiLSTM keypoint teacher. It is not a
reproduction of the paper's ST-GCN architecture, compression, quantization,
or MCU deployment. The objectives are classification cross-entropy, softened
teacher-to-student logit KL, and direct sequence-level cosine hint matching.
No temporal attention or teacher-state retrieval is used.

The MM-Fi folds and student architecture follow KD_Align.py. Train uses
paired mmWave and camera-derived 2D keypoints. Validation/test use radar
only. No random seed is set.

Example:
    python KD_Align_Shin_ResponseHint_Adapted.py \
        --config /path/to/config.yaml \
        --data-root /path/to/MMFi/Unzip \
        --teacher-root /path/to/rgb_baseline \
        --save-root /path/to/shin_response_hint_adapted
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from mmfi import MMFi_Dataset, make_dataloader, make_dataset
from PNL_reweight_Segment_Align import PointNetLSTM
from ResNet18_LSTM import GrayTeacherNet


ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def labels_from_batch(batch, device):
    return torch.tensor(
        [ACTION_TO_INDEX[action] for action in batch["action"]],
        dtype=torch.long,
        device=device,
    )


def radar_from_batch(batch, device):
    radar = batch["input_mmwave"].to(device=device, dtype=torch.float32)
    if radar.ndim != 4 or radar.shape[-1] != 5:
        raise ValueError(f"Expected radar (B,T,N,5), got {tuple(radar.shape)}")
    return radar.permute(0, 1, 3, 2).contiguous()


def keypoints_from_batch(batch, device):
    keypoints = batch["input_rgb"].to(device=device, dtype=torch.float32)
    if keypoints.ndim != 4 or keypoints.shape[-2:] != (17, 2):
        raise ValueError(
            f"Expected keypoints (B,T,17,2), got {tuple(keypoints.shape)}"
        )
    return keypoints


def response_kd_loss(student_logits, teacher_logits, temperature):
    """T^2 * KL(softmax(teacher/T) || softmax(student/T))."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_probabilities = F.softmax(
        teacher_logits / temperature, dim=-1
    )
    student_log_probabilities = F.log_softmax(
        student_logits / temperature, dim=-1
    )
    return F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * temperature**2


def cosine_hint_loss(student_segments, teacher_sequence):
    """Compare global hidden representations without temporal retrieval.

    Student segments are projected to the teacher's 1024-D latent space by
    PointNetLSTM.align_proj. Max pooling across K segments and T teacher
    states matches the respective encoders' global max-pooling convention.
    """
    student_global = student_segments.max(dim=1).values
    teacher_global = teacher_sequence.max(dim=1).values
    if student_global.shape != teacher_global.shape:
        raise ValueError(
            "Hint dimensions must match; expected student align_dim=1024 "
            "and teacher temporal dimension=1024"
        )
    return (1.0 - F.cosine_similarity(
        student_global, teacher_global, dim=-1, eps=1e-8
    )).mean()


def load_teacher(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing fold-matched teacher: {checkpoint_path}")
    teacher = GrayTeacherNet(num_classes=27, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported teacher checkpoint: {checkpoint_path}")
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }
    # A logit teacher requires its complete trained classifier as well as
    # its CNN and LSTM. Fail rather than silently use random parameters.
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def teacher_forward_chunked(teacher, keypoints, frame_chunk_size):
    """Exact eval-mode teacher forward with smaller CNN frame batches."""
    batch_size, time_steps, _, _ = keypoints.shape
    frames = keypoints.reshape(-1, 1, 17, 2)
    cnn_chunks = [teacher.cnn(chunk) for chunk in frames.split(
        frame_chunk_size, dim=0
    )]
    frame_features = torch.cat(cnn_chunks, dim=0)
    frame_features = frame_features.reshape(batch_size, time_steps, 512)
    teacher_sequence, _ = teacher.lstm(frame_features)
    teacher_global = teacher_sequence.max(dim=1).values
    teacher_logits = teacher.classifier(teacher_global)
    return teacher_logits, teacher_sequence


def train_one_epoch(student, teacher, loader, optimizer, device,
                    ce_weight, response_weight, hint_weight,
                    temperature, frame_chunk_size):
    student.train()
    teacher.eval()
    totals = np.zeros(4, dtype=np.float64)
    sample_count = 0
    for batch in tqdm(loader, desc="Response+hint training", leave=False):
        radar = radar_from_batch(batch, device)
        keypoints = keypoints_from_batch(batch, device)
        labels = labels_from_batch(batch, device)

        teacher_logits, teacher_sequence = teacher_forward_chunked(
            teacher, keypoints, frame_chunk_size
        )
        student_logits, student_segments = student(
            radar, return_features=True
        )
        ce_loss = F.cross_entropy(student_logits, labels)
        response_loss = response_kd_loss(
            student_logits, teacher_logits, temperature
        )
        hint_loss = cosine_hint_loss(
            student_segments, teacher_sequence
        )
        total_loss = (
            ce_weight * ce_loss
            + response_weight * response_loss
            + hint_weight * hint_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite training loss")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        sample_count += batch_size
        totals += np.asarray([
            total_loss.item(),
            ce_loss.item(),
            response_loss.item(),
            hint_loss.item(),
        ]) * batch_size
    if sample_count == 0:
        raise RuntimeError("Empty train loader; check the split and batch size")
    return (totals / sample_count).tolist()


def evaluate(student, loader, device):
    student.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Radar-only eval", leave=False):
            logits = student(radar_from_batch(batch, device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(labels_from_batch(batch, device).cpu().tolist())
    if not labels:
        raise RuntimeError("Empty validation/test loader")
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def run(args):
    with args.config.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if config.get("split_to_use") != "kfold_cross_subject":
        raise ValueError("Expected split_to_use: kfold_cross_subject")
    if config.get("data_unit") != "sequence":
        raise ValueError("Expected data_unit: sequence")
    modalities = set(config.get("modality", "").split("|"))
    if not {"mmwave", "rgb"}.issubset(modalities):
        raise ValueError("config modality must include mmwave|rgb")
    if not 1 <= args.num_segments <= 297:
        raise ValueError("num-segments must be between 1 and 297")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; no random seed is set")
    print("Validation/test modality: mmwave only")
    fold_results = []

    for fold in range(config["kfold"]["num_folds"]):
        print(f"\n=== Shin et al. response+hint adaptation / fold {fold} ===")
        config["kfold"]["current_fold"] = fold
        train_set, validation_multimodal, test_multimodal = make_dataset(
            str(args.data_root), config
        )
        validation_set = MMFi_Dataset(
            train_set.data_base, "sequence", "mmwave", "validation",
            validation_multimodal.data_source,
        )
        test_set = MMFi_Dataset(
            train_set.data_base, "sequence", "mmwave", "testing",
            test_multimodal.data_source,
        )
        train_loader = make_dataloader(
            train_set, True, generator=None, batch_size=args.batch_size
        )
        validation_loader = make_dataloader(
            validation_set, False, generator=None,
            batch_size=args.eval_batch_size,
        )
        test_loader = make_dataloader(
            test_set, False, generator=None,
            batch_size=args.eval_batch_size,
        )
        teacher = load_teacher(
            args.teacher_root / f"fold_{fold}" / "best_res18_pretrain.pth",
            device,
        )
        student = PointNetLSTM(
            num_classes=27,
            num_segments=args.num_segments,
            align_dim=1024,
        ).to(device)
        optimizer = torch.optim.Adam(
            student.parameters(), lr=args.learning_rate
        )
        fold_dir = args.save_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_path = fold_dir / "best_shin_response_hint_adapted.pth"
        best_validation_accuracy = float("-inf")

        for epoch in range(1, args.epochs + 1):
            total, ce, response, hint = train_one_epoch(
                student, teacher, train_loader, optimizer, device,
                args.ce_weight, args.response_weight, args.hint_weight,
                args.temperature, args.frame_chunk_size,
            )
            val_accuracy, val_f1 = evaluate(
                student, validation_loader, device
            )
            print(
                f"[Fold {fold}] epoch {epoch:03d} loss={total:.4f} "
                f"CE={ce:.4f} response={response:.4f} hint={hint:.4f} "
                f"val_acc={val_accuracy:.4f} val_macro_f1={val_f1:.4f}"
            )
            if val_accuracy > best_validation_accuracy:
                best_validation_accuracy = val_accuracy
                torch.save({
                    "model_state_dict": student.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": val_accuracy,
                    "validation_macro_f1": val_f1,
                    "method": "Shin et al. response+hint; common backbone",
                }, best_path)

        best = torch.load(best_path, map_location=device)
        student.load_state_dict(best["model_state_dict"], strict=True)
        test_accuracy, test_f1 = evaluate(student, test_loader, device)
        fold_result = {
            "fold": fold,
            "best_epoch": best["epoch"],
            "best_validation_accuracy": best["validation_accuracy"],
            "best_validation_macro_f1": best["validation_macro_f1"],
            "test_acc": test_accuracy,
            "test_f1": test_f1,
        }
        fold_results.append(fold_result)
        print(f"[Fold {fold}] test Acc={test_accuracy:.4f} F1={test_f1:.4f}")
        (fold_dir / "result.json").write_text(
            json.dumps(fold_result, indent=2), encoding="utf-8"
        )

    accuracies = np.asarray([result["test_acc"] for result in fold_results])
    f1_scores = np.asarray([result["test_f1"] for result in fold_results])
    summary = {
        "method": "Shin et al. response+hint common-backbone adaptation",
        "random_seed_fixed": False,
        "evaluation_modality": "mmwave only",
        "test_subjects_fixed_across_folds": True,
        "subjects_test": config["kfold"]["subjects_test"],
        "subjects_trainval": config["kfold"]["subjects_trainval"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_segments": args.num_segments,
            "temperature": args.temperature,
            "ce_weight": args.ce_weight,
            "response_weight": args.response_weight,
            "hint_weight": args.hint_weight,
            "frame_chunk_size": args.frame_chunk_size,
        },
        "mean_test_acc": float(accuracies.mean()),
        "std_test_acc": float(accuracies.std(ddof=1)),
        "mean_test_f1": float(f1_scores.mean()),
        "std_test_f1": float(f1_scores.std(ddof=1)),
        "folds": fold_results,
    }
    summary_path = args.save_root / "shin_response_hint_adapted_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== FINAL SHIN RESPONSE+HINT ADAPTED RESULT ===")
    print(f"Test Acc: {summary['mean_test_acc']:.4f} +/- {summary['std_test_acc']:.4f}")
    print(f"Test F1 : {summary['mean_test_f1']:.4f} +/- {summary['std_test_f1']:.4f}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=0.5)
    parser.add_argument("--hint-weight", type=float, default=0.5)
    parser.add_argument("--frame-chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1:
        parser.error("epochs>=1, batch-size>=2, eval-batch-size>=1 required")
    if args.learning_rate <= 0 or args.temperature <= 0:
        parser.error("learning-rate and temperature must be positive")
    if args.frame_chunk_size < 1:
        parser.error("frame-chunk-size must be positive")
    if min(args.ce_weight, args.response_weight, args.hint_weight) < 0:
        parser.error("loss weights must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
