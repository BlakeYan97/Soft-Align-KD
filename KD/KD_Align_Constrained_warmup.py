"""Soft temporal alignment KD with anti-collapse constraints and diagnostics.

Place this file beside mmfi.py, PNL_reweight_Segment_Align.py and
ResNet18_LSTM.py.  The original model definitions do not need to change.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from mmfi import make_dataloader, make_dataset
from PNL_reweight_Segment_Align import PointNetLSTM
from ResNet18_LSTM import GrayTeacherNet


ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
ACTION_TO_IDX = {action: index for index, action in enumerate(ACTIONS)}


def actions_to_labels(action_list):
    return torch.tensor([ACTION_TO_IDX[action] for action in action_list], dtype=torch.long)


def preprocess_mmwave(batch, device):
    """Convert (B,T,N,D) mmWave points to PointNet's (B,T,D,N)."""
    x = batch.get("input_mmwave")
    if x is None:
        raise RuntimeError("Batch missing input_mmwave")
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.float32).permute(0, 1, 3, 2).contiguous()


def attention_diagnostics(attn):
    """Differentiable summaries, except duplicate_rate which is for reporting."""
    batch_size, num_segments, teacher_steps = attn.shape
    time = torch.linspace(0.0, 1.0, teacher_steps, device=attn.device, dtype=attn.dtype)
    centers = torch.sum(attn * time.view(1, 1, -1), dim=-1)
    gaps = centers[:, 1:] - centers[:, :-1]

    unit_attn = F.normalize(attn, p=2, dim=-1, eps=1e-8)
    gram = torch.bmm(unit_attn, unit_attn.transpose(1, 2))
    off_diag = ~torch.eye(num_segments, device=attn.device, dtype=torch.bool)
    overlap = gram[:, off_diag].mean()

    peaks = attn.argmax(dim=-1)
    upper_triangle = torch.triu(
        torch.ones(num_segments, num_segments, device=attn.device, dtype=torch.bool),
        diagonal=1,
    )
    peak_matches = peaks.unsqueeze(2).eq(peaks.unsqueeze(1))
    duplicate_rate = peak_matches[:, upper_triangle].float().mean()
    entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = (entropy / math.log(teacher_steps)).mean()

    return {
        "centers": centers,
        "gaps": gaps,
        "overlap": overlap,
        "duplicate_rate": duplicate_rate,
        "order_violation_rate": (gaps <= 0.0).float().mean(),
        "normalized_entropy": entropy,
    }


def soft_alignment_kd_loss(
    student_seg,
    teacher_seq,
    tau=0.08,
    min_gap_ratio=0.5,
    diversity_weight=0.1,
    alignment_mode="original",
):
    """Soft alignment plus an order-preserving, anti-collapse constraint.

    ``alignment_mode='original'`` preserves the exact attention and KD equations
    from KD_Align.py, making the constraint experiment a fair one-factor ablation.
    ``alignment_mode='cosine'`` retains the earlier experimental implementation.
    """
    if student_seg.ndim != 3 or teacher_seq.ndim != 3:
        raise ValueError("Expected student_seg=(B,K,D) and teacher_seq=(B,T,D)")
    if student_seg.shape[0] != teacher_seq.shape[0] or student_seg.shape[2] != teacher_seq.shape[2]:
        raise ValueError(
            f"Incompatible alignment shapes: {tuple(student_seg.shape)} vs "
            f"{tuple(teacher_seq.shape)}"
        )
    if tau <= 0:
        raise ValueError("tau must be positive")

    student_norm = F.normalize(student_seg, dim=-1)
    teacher_norm = F.normalize(teacher_seq, dim=-1)
    if alignment_mode == "original":
        similarity = torch.einsum("bkd,btd->bkt", student_seg, teacher_seq) / tau
    elif alignment_mode == "cosine":
        similarity = torch.einsum("bkd,btd->bkt", student_norm, teacher_norm) / tau
    else:
        raise ValueError(f"Unknown alignment_mode: {alignment_mode}")
    attn = torch.softmax(similarity, dim=-1)

    if alignment_mode == "original":
        teacher_aligned = torch.einsum("bkt,btd->bkd", attn, teacher_seq)
        kd_loss = F.mse_loss(student_norm, F.normalize(teacher_aligned, dim=-1))
    else:
        teacher_aligned = torch.einsum("bkt,btd->bkd", attn, teacher_norm)
        teacher_aligned = F.normalize(teacher_aligned, dim=-1)
        kd_loss = (1.0 - F.cosine_similarity(student_norm, teacher_aligned, dim=-1)).mean()

    diagnostics = attention_diagnostics(attn)
    num_segments = student_seg.shape[1]
    min_gap = min_gap_ratio / num_segments
    # Division by min_gap makes a complete center collapse contribute about 1.
    order_loss = (F.relu(min_gap - diagnostics["gaps"]) / min_gap).square().mean()
    diversity_loss = diagnostics["overlap"]
    constraint_loss = order_loss + diversity_weight * diversity_loss

    diagnostics.update({
        "order_loss": order_loss,
        "diversity_loss": diversity_loss,
        "constraint_loss": constraint_loss,
    })
    return kd_loss, constraint_loss, attn, diagnostics


def _mean_dict(rows):
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


class DistillationTrainer:
    def __init__(
        self,
        teacher,
        student,
        optimizer,
        device,
        kd_weight=0.5,
        constraint_weight=0.2,
        tau=0.08,
        min_gap_ratio=0.5,
        diversity_weight=0.1,
        alignment_mode="original",
        constraint_start_epoch=10,
        constraint_warmup_epochs=10,
        max_grad_norm=5.0,
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.optimizer = optimizer
        self.device = device
        self.kd_weight = kd_weight
        self.max_constraint_weight = constraint_weight
        self.constraint_weight = constraint_weight
        self.tau = tau
        self.min_gap_ratio = min_gap_ratio
        self.diversity_weight = diversity_weight
        self.alignment_mode = alignment_mode
        self.constraint_start_epoch = constraint_start_epoch
        self.constraint_warmup_epochs = constraint_warmup_epochs
        self.max_grad_norm = max_grad_norm

    def set_epoch(self, epoch):
        """Delay, then linearly introduce the auxiliary anti-collapse loss."""
        if epoch <= self.constraint_start_epoch:
            factor = 0.0
        elif self.constraint_warmup_epochs <= 0:
            factor = 1.0
        else:
            factor = min(
                1.0,
                (epoch - self.constraint_start_epoch) / self.constraint_warmup_epochs,
            )
        self.constraint_weight = self.max_constraint_weight * factor

    def _alignment_loss(self, student_features, teacher_features):
        return soft_alignment_kd_loss(
            student_features,
            teacher_features,
            tau=self.tau,
            min_gap_ratio=self.min_gap_ratio,
            diversity_weight=self.diversity_weight,
            alignment_mode=self.alignment_mode,
        )

    def train_one_epoch(self, loader):
        self.student.train()
        rows = []
        for batch in tqdm(loader, desc="Training", leave=False):
            mmwave = preprocess_mmwave(batch, self.device)
            rgb = batch["input_rgb"].unsqueeze(2).float().to(self.device)
            labels = actions_to_labels(batch["action"]).to(self.device)

            with torch.no_grad():
                _, teacher_features = self.teacher(rgb, return_features=True)
            logits, student_features = self.student(mmwave, return_features=True)

            ce_loss = F.cross_entropy(logits, labels)
            kd_loss, constraint_loss, _, diagnostics = self._alignment_loss(
                student_features, teacher_features
            )
            total_loss = (
                ce_loss
                + self.kd_weight * kd_loss
                + self.constraint_weight * constraint_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()

            rows.append({
                "loss": total_loss.item(),
                "ce": ce_loss.item(),
                "kd": kd_loss.item(),
                "constraint": constraint_loss.item(),
                "duplicate_rate": diagnostics["duplicate_rate"].item(),
                "order_violation_rate": diagnostics["order_violation_rate"].item(),
            })
        return _mean_dict(rows)

    def evaluate(self, loader):
        self.student.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                mmwave = preprocess_mmwave(batch, self.device)
                labels = actions_to_labels(batch["action"]).to(self.device)
                predictions.extend(self.student(mmwave).argmax(dim=1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
        return (
            accuracy_score(targets, predictions),
            f1_score(targets, predictions, average="macro"),
        )

    def export_alignment_evidence(self, loader, output_dir, max_batches=None, max_plots=8):
        """Save paper-ready raw attention, metrics, and representative heatmaps."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.student.eval()
        all_attn, records, labels = [], [], []
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(loader, desc="Alignment analysis", leave=False)):
                if max_batches is not None and batch_index >= max_batches:
                    break
                mmwave = preprocess_mmwave(batch, self.device)
                rgb = batch["input_rgb"].unsqueeze(2).float().to(self.device)
                _, teacher_features = self.teacher(rgb, return_features=True)
                _, student_features = self.student(mmwave, return_features=True)
                _, _, attn, diagnostics = self._alignment_loss(student_features, teacher_features)
                all_attn.append(attn.cpu())

                centers = diagnostics["centers"].cpu()
                gaps = diagnostics["gaps"].cpu()
                for index, action in enumerate(batch["action"]):
                    peaks = attn[index].argmax(dim=-1).cpu()
                    records.append({
                        "action": action,
                        "centers": centers[index].tolist(),
                        "adjacent_gaps": gaps[index].tolist(),
                        "peak_frames_zero_based": peaks.tolist(),
                        "duplicate_rate": float(
                            peaks.unsqueeze(1).eq(peaks.unsqueeze(0))[
                                torch.triu(torch.ones(peaks.numel(), peaks.numel(), dtype=torch.bool), diagonal=1)
                            ].float().mean()
                        ),
                        "order_violation_rate": float((gaps[index] <= 0).float().mean()),
                    })
                    labels.append(action)

        if not all_attn:
            raise RuntimeError("No batches were available for alignment analysis")
        os.makedirs(output_dir, exist_ok=True)
        attention = torch.cat(all_attn, dim=0).numpy()
        np.savez_compressed(os.path.join(output_dir, "attention_weights.npz"), attention=attention)
        attention_unit = attention / np.maximum(
            np.linalg.norm(attention, axis=-1, keepdims=True), 1e-12
        )
        attention_gram = attention_unit @ attention_unit.transpose(0, 2, 1)
        off_diagonal = ~np.eye(attention.shape[1], dtype=bool)

        aggregate = {
            "num_samples": len(records),
            "mean_duplicate_rate": float(np.mean([x["duplicate_rate"] for x in records])),
            "mean_order_violation_rate": float(np.mean([x["order_violation_rate"] for x in records])),
            "mean_adjacent_gap": float(np.mean([x["adjacent_gaps"] for x in records])),
            "mean_attention_overlap": float(attention_gram[:, off_diagonal].mean()),
            "mean_normalized_entropy": float(
                np.mean(
                    -(np.clip(attention, 1e-12, None) * np.log(np.clip(attention, 1e-12, None))).sum(-1)
                    / np.log(attention.shape[-1])
                )
            ),
            "samples": records,
        }
        with open(os.path.join(output_dir, "alignment_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)

        count = min(max_plots, len(attention))
        columns = min(4, count)
        rows = math.ceil(count / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.8 * rows), squeeze=False)
        image = None
        for index, axis in enumerate(axes.flat):
            if index >= count:
                axis.axis("off")
                continue
            image = axis.imshow(attention[index], aspect="auto", origin="lower", cmap="viridis")
            peaks = attention[index].argmax(axis=-1)
            axis.scatter(peaks, np.arange(attention.shape[1]), c="red", s=18, marker="x")
            axis.set_title(f"Sample {index + 1} / {labels[index]}")
            axis.set_xlabel("Teacher frame")
            axis.set_ylabel("Student segment")
            axis.set_yticks(np.arange(attention.shape[1]))
            axis.set_yticklabels(np.arange(1, attention.shape[1] + 1))
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="Attention weight", shrink=0.85)
        figure.suptitle("Order-constrained soft temporal alignment", fontsize=14)
        figure.subplots_adjust(left=0.07, right=0.9, bottom=0.1, top=0.88, wspace=0.28, hspace=0.4)
        figure.savefig(os.path.join(output_dir, "attention_heatmaps.png"), dpi=300)
        figure.savefig(os.path.join(output_dir, "attention_heatmaps.pdf"), bbox_inches="tight")
        plt.close(figure)

        mean_attention = attention.mean(axis=0)
        figure, axis = plt.subplots(figsize=(10, 3.2))
        image = axis.imshow(mean_attention, aspect="auto", origin="lower", cmap="viridis")
        peaks = mean_attention.argmax(axis=-1)
        axis.scatter(peaks, np.arange(mean_attention.shape[0]), c="red", s=30, marker="x")
        axis.set_xlabel("Teacher frame")
        axis.set_ylabel("Student segment")
        axis.set_yticks(np.arange(mean_attention.shape[0]))
        axis.set_yticklabels(np.arange(1, mean_attention.shape[0] + 1))
        axis.set_title(f"Mean attention over all {len(attention)} test samples")
        figure.colorbar(image, ax=axis, label="Mean attention weight")
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, "mean_attention_heatmap.png"), dpi=300)
        figure.savefig(os.path.join(output_dir, "mean_attention_heatmap.pdf"), bbox_inches="tight")
        plt.close(figure)
        return aggregate


def run_distillation(
    config_path,
    data_root,
    teacher_root,
    save_root,
    student_init_root=None,
):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    num_folds = config["kfold"]["num_folds"]
    alignment_cfg = config.get("soft_alignment", {})
    epochs = alignment_cfg.get("epochs", config.get("epochs", 50))
    folds_to_run = alignment_cfg.get("folds", list(range(num_folds)))
    if not folds_to_run:
        raise ValueError("soft_alignment.folds must contain at least one fold index")
    if any(fold < 0 or fold >= num_folds for fold in folds_to_run):
        raise ValueError(f"Invalid fold list {folds_to_run}; num_folds={num_folds}")
    os.makedirs(save_root, exist_ok=True)
    fold_results = []

    for fold in folds_to_run:
        print(f"\n=== KD TRAINING - Fold {fold} ===")
        config["kfold"]["current_fold"] = fold
        fold_seed = config.get("seed", 0) + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        generator = torch.Generator().manual_seed(fold_seed)
        train_ds, val_ds, test_ds = make_dataset(data_root, config)
        train_loader = make_dataloader(train_ds, True, generator=generator, batch_size=32)
        val_loader = make_dataloader(val_ds, False, generator=generator, batch_size=16)
        test_loader = make_dataloader(test_ds, False, generator=generator, batch_size=16)

        teacher = GrayTeacherNet(num_classes=27).to(device)
        teacher_path = os.path.join(teacher_root, f"fold_{fold}", "best_res18_pretrain.pth")
        teacher.load_state_dict(torch.load(teacher_path, map_location=device), strict=False)

        student = PointNetLSTM(num_classes=27).to(device)
        if student_init_root is not None:
            init_path = os.path.join(
                student_init_root, f"fold_{fold}", "best_student_softalign.pth"
            )
            if not os.path.isfile(init_path):
                raise FileNotFoundError(f"Original Student checkpoint not found: {init_path}")
            init_checkpoint = torch.load(init_path, map_location=device)
            init_state = init_checkpoint.get("model_state_dict", init_checkpoint)
            student.load_state_dict(init_state, strict=True)
            print(f"Fine-tuning original KD checkpoint: {init_path}")

        learning_rate = alignment_cfg.get(
            "learning_rate", 1e-5 if student_init_root is not None else 1e-4
        )
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            device=device,
            kd_weight=alignment_cfg.get("kd_weight", 0.5),
            constraint_weight=alignment_cfg.get("constraint_weight", 0.05),
            tau=alignment_cfg.get("tau", 0.08),
            min_gap_ratio=alignment_cfg.get("min_gap_ratio", 0.5),
            diversity_weight=alignment_cfg.get("diversity_weight", 0.1),
            alignment_mode=alignment_cfg.get("alignment_mode", "original"),
            constraint_start_epoch=alignment_cfg.get("constraint_start_epoch", 10),
            constraint_warmup_epochs=alignment_cfg.get("constraint_warmup_epochs", 10),
            max_grad_norm=alignment_cfg.get("max_grad_norm", 5.0),
        )

        if student_init_root is not None:
            initial_val_accuracy, initial_val_f1 = trainer.evaluate(val_loader)
            print(
                f"[Fold {fold}] Before fine-tuning: val_acc={initial_val_accuracy:.4f} "
                f"val_f1={initial_val_f1:.4f}"
            )
        else:
            print(f"[Fold {fold}] Training from scratch with seed {fold_seed}")

        best_acc = -1.0
        min_checkpoint_epoch = alignment_cfg.get(
            "min_checkpoint_epoch",
            trainer.constraint_start_epoch + trainer.constraint_warmup_epochs,
        )
        fold_dir = os.path.join(save_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model_path = os.path.join(fold_dir, "best_student_constrained_softalign.pth")
        for epoch in range(1, epochs + 1):
            trainer.set_epoch(epoch)
            train_metrics = trainer.train_one_epoch(train_loader)
            accuracy, macro_f1 = trainer.evaluate(val_loader)
            print(
                f"[Fold {fold}] Epoch {epoch}: loss={train_metrics['loss']:.4f} "
                f"ce={train_metrics['ce']:.4f} kd={train_metrics['kd']:.4f} "
                f"constraint={train_metrics['constraint']:.4f} "
                f"constraint_weight={trainer.constraint_weight:.4f} "
                f"duplicate={train_metrics['duplicate_rate']:.4f} "
                f"order_violation={train_metrics['order_violation_rate']:.4f} "
                f"val_acc={accuracy:.4f} val_f1={macro_f1:.4f}"
            )
            # Exclude early checkpoints that have not yet seen the full
            # anti-collapse objective.
            if epoch >= min_checkpoint_epoch and accuracy > best_acc:
                best_acc = accuracy
                torch.save({
                    "model_state_dict": student.state_dict(),
                    "soft_alignment": alignment_cfg,
                    "epoch": epoch,
                    "val_accuracy": accuracy,
                }, model_path)

        checkpoint = torch.load(model_path, map_location=device)
        student.load_state_dict(checkpoint["model_state_dict"])
        test_accuracy, test_f1 = trainer.evaluate(test_loader)
        evidence = trainer.export_alignment_evidence(
            test_loader, os.path.join(fold_dir, "alignment_evidence")
        )
        result = {
            "fold": fold,
            "test_acc": test_accuracy,
            "test_f1": test_f1,
            "attention_duplicate_rate": evidence["mean_duplicate_rate"],
            "attention_order_violation_rate": evidence["mean_order_violation_rate"],
            "attention_mean_adjacent_gap": evidence["mean_adjacent_gap"],
            "attention_overlap": evidence["mean_attention_overlap"],
            "attention_normalized_entropy": evidence["mean_normalized_entropy"],
        }
        fold_results.append(result)
        print(f"[Fold {fold}] {result}")

    summary = {
        "mean_test_acc": float(np.mean([x["test_acc"] for x in fold_results])),
        "std_test_acc": float(np.std([x["test_acc"] for x in fold_results], ddof=1)) if len(fold_results) > 1 else 0.0,
        "mean_test_f1": float(np.mean([x["test_f1"] for x in fold_results])),
        "std_test_f1": float(np.std([x["test_f1"] for x in fold_results], ddof=1)) if len(fold_results) > 1 else 0.0,
        "mean_attention_duplicate_rate": float(np.mean([x["attention_duplicate_rate"] for x in fold_results])),
        "mean_attention_order_violation_rate": float(np.mean([x["attention_order_violation_rate"] for x in fold_results])),
        "mean_attention_overlap": float(np.mean([x["attention_overlap"] for x in fold_results])),
        "mean_attention_normalized_entropy": float(np.mean([x["attention_normalized_entropy"] for x in fold_results])),
        "folds": fold_results,
    }
    with open(os.path.join(save_root, "kd_constrained_softalign_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("\n=== FINAL CONSTRAINED KD RESULT ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_distillation(
        config_path="config.yaml",
        data_root="data/MMFi",
        teacher_root="checkpoints/teacher",
        save_root="outputs/soft_align_constrained",
        student_init_root=None,
    )
