"""Common-backbone, 2D-pose adaptation of Tac-Mamba on MM-Fi.

This is an adapted comparison baseline, NOT a reproduction of the full
Tac-Mamba model. It replaces the paper's point/visual/Mamba encoders with
the project's PointNet, frozen ResNet-18, and Bi-LSTM encoders. It retains
radar-query/visual-key-value cross-attention, trust-aware gating, visual
modality dropout, and auxiliary pose prediction. The paper uses a 3D pose
target; this adaptation uses the same camera-derived 2D keypoints available
to the project's teacher and normalizes them per frame.

During validation and testing, the visual input is absent. The saved
checkpoint is a radar-only model; no teacher weights are stored in it.

Example:
    python KD_Align_TacMamba_Adapted.py \
        --config /path/to/config.yaml \
        --data-root /path/to/MMFi/Unzip \
        --teacher-root /path/to/rgb_baseline \
        --save-root /path/to/tac_mamba_adapted

The config must use data_unit: sequence and include both mmwave and rgb.
No random seed is set. As in the existing code, the five development folds
are subject-disjoint and the same fixed test subjects are used for each run.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from mmfi import MMFi_Dataset, make_dataset, make_dataloader
from PNL_reweight_Segment_Align import (
    PointNetfeat, PostReweight, segment_max_pooling, resolve_point_mask,
)
from ResNet18_LSTM import GrayTeacherNet


ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTIONS)}


def labels_from_batch(batch, device):
    return torch.tensor(
        [ACTION_TO_IDX[name] for name in batch["action"]],
        dtype=torch.long,
        device=device,
    )


def radar_from_batch(batch, device):
    radar = batch["input_mmwave"].to(device=device, dtype=torch.float32)
    if radar.ndim != 4 or radar.shape[-1] != 5:
        raise ValueError(f"Expected radar (B,T,N,5), got {tuple(radar.shape)}")
    return radar.permute(0, 1, 3, 2).contiguous()


def visual_from_batch(batch, device):
    keypoints = batch["input_rgb"].to(device=device, dtype=torch.float32)
    if keypoints.ndim != 4 or keypoints.shape[-2:] != (17, 2):
        raise ValueError(
            f"Expected camera keypoints (B,T,17,2), got {tuple(keypoints.shape)}"
        )
    return keypoints


def relative_2d_pose(keypoints):
    """Center each frame and divide by its RMS joint distance (unit-free)."""
    centered = keypoints - keypoints.mean(dim=-2, keepdim=True)
    scale = centered.square().sum(dim=-1).mean(dim=-1, keepdim=True)
    scale = scale.clamp_min(1e-12).sqrt().unsqueeze(-1)
    return centered / scale


class FrozenVisualEncoder(nn.Module):
    """Use the pretrained teacher's frame CNN; do not update its weights."""

    def __init__(self, teacher):
        super().__init__()
        self.cnn = teacher.cnn
        self.cnn.requires_grad_(False)
        self.cnn.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, keypoints, frame_chunk_size=128):
        batch_size, time_steps, _, _ = keypoints.shape
        frames = keypoints.reshape(-1, 1, 17, 2)
        encoded = []
        for chunk in frames.split(frame_chunk_size, dim=0):
            encoded.append(self.cnn(chunk))
        return torch.cat(encoded, dim=0).reshape(batch_size, time_steps, 512)


class RadarVisualGatedAttention(nn.Module):
    """Tac-Mamba-inspired cross-attention and trust-aware visual gating.

    Visual features are masked AFTER their projection, making K and V
    exactly zero when the camera modality is absent. In this case the
    cross-attention output is zero and the learned radar fallback remains.
    """

    def __init__(self, radar_dim=1024, visual_dim=512, attention_dim=128):
        super().__init__()
        self.attention_dim = attention_dim
        self.radar_proj = nn.Linear(radar_dim, attention_dim)
        self.visual_proj = nn.Linear(visual_dim, attention_dim)
        self.query = nn.Linear(attention_dim, attention_dim, bias=False)
        self.key = nn.Linear(attention_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, attention_dim, bias=False)
        self.radar_gate = nn.Linear(attention_dim, attention_dim)
        self.visual_gate = nn.Linear(attention_dim, attention_dim)
        self.fallback_gate = nn.Linear(attention_dim, attention_dim)
        self.back_proj = nn.Linear(attention_dim, radar_dim)

    def forward(self, radar_frames, visual_frames=None, visual_dropout=0.0):
        batch_size = radar_frames.shape[0]
        radar_small = self.radar_proj(radar_frames)

        if visual_frames is None:
            visual_small = torch.zeros_like(radar_small)
            # Attention to all-zero values is exactly zero. Avoid constructing
            # a T-by-T matrix during radar-only inference.
            retrieved = torch.zeros_like(radar_small)
        else:
            visual_small = self.visual_proj(visual_frames)
            if visual_dropout > 0:
                # Sample-level camera dropout, as a missing modality for an
                # entire activity sequence rather than isolated frames.
                keep = (torch.rand(batch_size, 1, 1, device=radar_frames.device)
                        >= visual_dropout).to(visual_small.dtype)
                visual_small = visual_small * keep

            query = self.query(radar_small)
            key = self.key(visual_small)
            value = self.value(visual_small)
            scores = query @ key.transpose(1, 2)
            scores = scores / math.sqrt(self.attention_dim)
            retrieved = torch.softmax(scores, dim=-1) @ value

        gate = torch.sigmoid(
            F.silu(self.radar_gate(radar_small))
            * F.silu(self.visual_gate(visual_small))
            + self.fallback_gate(radar_small)
        )
        fused_small = gate * retrieved + (1.0 - gate) * radar_small

        # The project's Bi-LSTM expects 1024-dimensional PointNet features.
        # A residual adapter preserves that interface while allowing the
        # cross-modal fusion to change the frame feature.
        return radar_frames + self.back_proj(fused_small - radar_small)


class TacMambaAdaptedStudent(nn.Module):
    """Original student backbone with an added frame-fusion adapter."""

    def __init__(self, num_classes=27, num_segments=4, attention_dim=128):
        super().__init__()
        self.pointnet = PointNetfeat()
        self.fusion = RadarVisualGatedAttention(attention_dim=attention_dim)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.num_segments = num_segments
        self.post_reweight = PostReweight(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.pose_head = nn.Linear(512, 34)

    def forward(self, radar, visual_frames=None, visual_dropout=0.0,
                return_pose=False, point_mask=None):
        point_mask = resolve_point_mask(radar, point_mask)
        batch_size, time_steps, _, _ = radar.shape
        frames = [self.pointnet(radar[:, t], point_mask=point_mask[:, t])
                  for t in range(time_steps)]
        radar_frames = torch.stack(frames, dim=1)

        if visual_frames is not None:
            if visual_frames.shape[:2] != (batch_size, time_steps):
                raise ValueError("Radar and visual sequence lengths must match")
        fused_frames = self.fusion(
            radar_frames, visual_frames, visual_dropout=visual_dropout
        )
        temporal_frames, _ = self.lstm(fused_frames)
        segments = segment_max_pooling(temporal_frames, self.num_segments)
        segments = self.post_reweight(segments)
        logits = self.classifier(segments.max(dim=1).values)

        if return_pose:
            pose = self.pose_head(temporal_frames).reshape(
                batch_size, time_steps, 17, 2
            )
            return logits, pose
        return logits


def evaluate(model, loader, device):
    """Camera-free validation/test, including checkpoint selection."""
    model.eval()
    predicted, actual = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Radar-only eval", leave=False):
            logits = model(radar_from_batch(batch, device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels_from_batch(batch, device).cpu().tolist())
    return (
        accuracy_score(actual, predicted),
        f1_score(actual, predicted, average="macro", zero_division=0),
    )


def train_one_epoch(model, visual_encoder, loader, optimizer, device,
                    pose_weight, visual_dropout, frame_chunk_size):
    model.train()
    visual_encoder.eval()
    totals = np.zeros(3, dtype=np.float64)
    batches = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        radar = radar_from_batch(batch, device)
        keypoints = visual_from_batch(batch, device)
        labels = labels_from_batch(batch, device)

        visual_frames = visual_encoder(keypoints, frame_chunk_size)
        logits, predicted_pose = model(
            radar,
            visual_frames=visual_frames,
            visual_dropout=visual_dropout,
            return_pose=True,
        )
        classification_loss = F.cross_entropy(logits, labels)
        pose_loss = F.mse_loss(predicted_pose, relative_2d_pose(keypoints))
        loss = classification_loss + pose_weight * pose_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals += [loss.item(), classification_loss.item(), pose_loss.item()]
        batches += 1
    if batches == 0:
        raise RuntimeError("Empty train loader; check the MM-Fi split and batch size")
    return (totals / batches).tolist()


def load_visual_encoder(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing fold-matched teacher: {checkpoint_path}")
    teacher = GrayTeacherNet(num_classes=27, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported teacher checkpoint: {checkpoint_path}")
    teacher.load_state_dict(checkpoint, strict=True)
    teacher.eval()
    return FrozenVisualEncoder(teacher)


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
        print(f"\n=== Tac-Mamba adapted / fold {fold} ===")
        config["kfold"]["current_fold"] = fold
        train_set, validation_multimodal, test_multimodal = make_dataset(
            str(args.data_root), config
        )
        validation_set = MMFi_Dataset(
            train_set.data_base,
            "sequence",
            "mmwave",
            "validation",
            validation_multimodal.data_source,
        )
        test_set = MMFi_Dataset(
            train_set.data_base,
            "sequence",
            "mmwave",
            "testing",
            test_multimodal.data_source,
        )
        # Passing None avoids the fixed default seed of torch.Generator().
        train_loader = make_dataloader(
            train_set, True, generator=None, batch_size=args.batch_size
        )
        validation_loader = make_dataloader(
            validation_set, False, generator=None, batch_size=args.eval_batch_size
        )
        test_loader = make_dataloader(
            test_set, False, generator=None, batch_size=args.eval_batch_size
        )
        visual_encoder = load_visual_encoder(
            args.teacher_root / f"fold_{fold}" / "best_res18_pretrain.pth",
            device,
        )
        model = TacMambaAdaptedStudent(
            num_classes=27,
            num_segments=args.num_segments,
            attention_dim=args.attention_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        fold_dir = args.save_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "best_tac_mamba_adapted.pth"
        best_validation_accuracy = float("-inf")

        for epoch in range(1, args.epochs + 1):
            total, classification, pose = train_one_epoch(
                model,
                visual_encoder,
                train_loader,
                optimizer,
                device,
                args.pose_weight,
                args.visual_dropout,
                args.frame_chunk_size,
            )
            val_accuracy, val_f1 = evaluate(model, validation_loader, device)
            print(
                f"[Fold {fold}] epoch {epoch:03d} "
                f"loss={total:.4f} CE={classification:.4f} "
                f"pose={pose:.4f} val_acc={val_accuracy:.4f} "
                f"val_macro_f1={val_f1:.4f}"
            )
            if val_accuracy > best_validation_accuracy:
                best_validation_accuracy = val_accuracy
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": val_accuracy,
                    "validation_macro_f1": val_f1,
                    "method": "Tac-Mamba-inspired; common backbone; 2D pose",
                }, checkpoint_path)

        best = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best["model_state_dict"], strict=True)
        test_accuracy, test_f1 = evaluate(model, test_loader, device)
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
        "method": "Tac-Mamba-inspired common-backbone adaptation (2D pose)",
        "test_subjects_fixed_across_folds": True,
        "subjects_test": config["kfold"]["subjects_test"],
        "subjects_trainval": config["kfold"]["subjects_trainval"],
        "random_seed_fixed": False,
        "evaluation_modality": "mmwave only",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_segments": args.num_segments,
            "attention_dim": args.attention_dim,
            "visual_dropout": args.visual_dropout,
            "pose_weight": args.pose_weight,
            "frame_chunk_size": args.frame_chunk_size,
        },
        "mean_test_acc": float(accuracies.mean()),
        "std_test_acc": float(accuracies.std(ddof=1)),
        "mean_test_f1": float(f1_scores.mean()),
        "std_test_f1": float(f1_scores.std(ddof=1)),
        "folds": fold_results,
    }
    summary_path = args.save_root / "tac_mamba_adapted_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== FINAL TAC-MAMBA ADAPTED RESULT ===")
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
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--visual-dropout", type=float, default=0.3)
    parser.add_argument("--pose-weight", type=float, default=0.01)
    parser.add_argument("--frame-chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1:
        parser.error("epochs>=1, batch-size>=2, eval-batch-size>=1 required")
    if not 0 <= args.visual_dropout < 1:
        parser.error("visual-dropout must be in [0,1)")
    if args.pose_weight < 0 or args.attention_dim < 1 or args.frame_chunk_size < 1:
        parser.error("pose-weight>=0, attention-dim>=1, frame-chunk-size>=1 required")
    return args


if __name__ == "__main__":
    run(parse_args())
