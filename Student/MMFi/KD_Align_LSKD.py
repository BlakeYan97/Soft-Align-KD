"""
MM-Fi cross-modal knowledge distillation with Logit Standardization (LS-KD).

Only the distillation objective is changed relative to KD_Align.py:
  - Teacher: the existing camera-derived 2D-keypoint GrayTeacherNet checkpoint.
  - Student: the existing PointNetLSTM with K=4 segment pooling and reweighting.
  - Split, data loaders, Adam optimizer, validation selection, and 5-fold test
    protocol follow the original project.

LS-KD source:
  Sun et al., "Logit Standardization in Knowledge Distillation", CVPR 2024.
The implementation of standardize_logits() and ls_kd_loss() follows the
authors' official implementation.
"""

import argparse
import json
import os
import random
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from mmfi import make_dataloader, make_dataset
from PNL_reweight_Segment_Align import PointNetLSTM
from ResNet18_LSTM import GrayTeacherNet


ACTIONS = ["A%02d" % i for i in range(1, 28)]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def actions_to_labels(action_list):
    return torch.tensor(
        [ACTION_TO_INDEX[action] for action in action_list],
        dtype=torch.long,
    )


def preprocess_mmwave(batch, device):
    """Convert mmWave input from (B,T,N,D) to (B,T,D,N)."""
    x = batch.get("input_mmwave")
    if x is None:
        raise RuntimeError("Batch is missing 'input_mmwave'.")
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return (
        x.to(device=device, dtype=torch.float32)
        .permute(0, 1, 3, 2)
        .contiguous()
    )


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_logits(logits: torch.Tensor, eps: float = 1e-7):
    """
    Per-sample Z-score standardization used by the official LS-KD code.

    PyTorch's default sample standard deviation (correction=1 / unbiased=True)
    is intentionally retained to match the official implementation.
    """
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + eps)


def ls_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
):
    """KL distillation after separately standardizing teacher/student logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    student_standardized = standardize_logits(student_logits)
    teacher_standardized = standardize_logits(teacher_logits)

    student_log_prob = F.log_softmax(
        student_standardized / temperature, dim=1
    )
    teacher_prob = F.softmax(
        teacher_standardized / temperature, dim=1
    )
    loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="none",
    ).sum(dim=1).mean()
    return loss * (temperature ** 2)


def _extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping/state_dict.")

    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break

    cleaned = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_teacher_checkpoint(model, checkpoint_path: str, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Teacher checkpoint does not exist: {checkpoint_path}"
        )

    raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(raw_checkpoint)
    model_state = model.state_dict()
    matched = {
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not matched:
        raise RuntimeError(
            "No teacher parameters matched the current GrayTeacherNet. "
            "Check the checkpoint format and model definition."
        )
    if "classifier.weight" not in matched:
        raise RuntimeError(
            "Teacher classifier.weight was not loaded; logit distillation "
            "would therefore be invalid."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded teacher checkpoint: {len(matched)}/{len(model_state)} "
        f"model tensors matched; missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )


class LSKDTrainer:
    def __init__(
        self,
        teacher,
        student,
        optimizer,
        device,
        ce_weight: float,
        kd_weight: float,
        temperature: float,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.optimizer = optimizer
        self.device = device
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight
        self.temperature = temperature

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train_one_epoch(self, loader):
        self.student.train()
        totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0}
        sample_count = 0

        for batch in tqdm(loader, desc="LS-KD training", leave=False):
            mmwave = preprocess_mmwave(batch, self.device)
            teacher_input = (
                batch["input_rgb"].unsqueeze(2).float().to(self.device)
            )
            labels = actions_to_labels(batch["action"]).to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(teacher_input)
            student_logits = self.student(mmwave)

            ce_loss = F.cross_entropy(student_logits, labels)
            kd_loss = ls_kd_loss(
                student_logits,
                teacher_logits,
                temperature=self.temperature,
            )
            loss = self.ce_weight * ce_loss + self.kd_weight * kd_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite LS-KD loss encountered. "
                    "Inspect logits and hyperparameters."
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["ce"] += ce_loss.item() * batch_size
            totals["kd"] += kd_loss.item() * batch_size

        if sample_count == 0:
            raise RuntimeError("The training loader produced no samples.")
        return {key: value / sample_count for key, value in totals.items()}

    def evaluate(self, loader):
        self.student.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluation", leave=False):
                mmwave = preprocess_mmwave(batch, self.device)
                labels = actions_to_labels(batch["action"]).to(self.device)
                logits = self.student(mmwave)
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(labels.cpu().tolist())

        if not targets:
            raise RuntimeError("The evaluation loader produced no samples.")
        accuracy = accuracy_score(targets, predictions)
        macro_f1 = f1_score(
            targets,
            predictions,
            labels=list(range(len(ACTIONS))),
            average="macro",
            zero_division=0,
        )
        return float(accuracy), float(macro_f1)


def run_distillation(
    config_path: str,
    data_root: str,
    teacher_root: str,
    save_root: str,
    device_string: Optional[str] = None,
    epochs_override: Optional[int] = None,
    learning_rate: float = 1e-4,
    ce_weight: float = 1.0,
    kd_weight: float = 9.0,
    temperature: float = 2.0,
    seed: int = 42,
):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(
        device_string
        if device_string is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    num_folds = int(config["kfold"]["num_folds"])
    epochs = (
        int(epochs_override)
        if epochs_override is not None
        else int(config.get("epochs", 50))
    )
    os.makedirs(save_root, exist_ok=True)

    hyperparameters: Dict[str, object] = {
        "method": "LS-KD",
        "ce_weight": ce_weight,
        "kd_weight": kd_weight,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "epochs": epochs,
        "train_batch_size": 32,
        "eval_batch_size": 16,
        "seed": seed,
        "logit_standardization_eps": 1e-7,
    }
    print("LS-KD settings:", json.dumps(hyperparameters, indent=2))

    fold_results = []
    for fold in range(num_folds):
        print("\n===============================")
        print(f"LS-KD TRAINING - Fold {fold}")
        print("===============================")

        fold_seed = seed + fold
        set_random_seed(fold_seed)
        generator = torch.Generator().manual_seed(fold_seed)
        config["kfold"]["current_fold"] = fold

        train_dataset, validation_dataset, test_dataset = make_dataset(
            data_root, config
        )
        train_loader = make_dataloader(
            train_dataset, True, generator=generator, batch_size=32
        )
        validation_loader = make_dataloader(
            validation_dataset, False, generator=generator, batch_size=16
        )
        test_loader = make_dataloader(
            test_dataset, False, generator=generator, batch_size=16
        )

        teacher = GrayTeacherNet(num_classes=len(ACTIONS)).to(device)
        teacher_checkpoint = os.path.join(
            teacher_root,
            f"fold_{fold}",
            "best_res18_pretrain.pth",
        )
        load_teacher_checkpoint(teacher, teacher_checkpoint, device)

        student = PointNetLSTM(num_classes=len(ACTIONS)).to(device)
        optimizer = torch.optim.Adam(
            student.parameters(), lr=learning_rate
        )
        trainer = LSKDTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            device=device,
            ce_weight=ce_weight,
            kd_weight=kd_weight,
            temperature=temperature,
        )

        fold_directory = os.path.join(save_root, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)
        best_checkpoint_path = os.path.join(
            fold_directory, "best_student_lskd.pth"
        )
        history_path = os.path.join(fold_directory, "history_lskd.json")

        best_validation_accuracy = -1.0
        best_validation_f1 = -1.0
        best_epoch = -1
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = trainer.train_one_epoch(train_loader)
            validation_accuracy, validation_f1 = trainer.evaluate(
                validation_loader
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce"],
                "train_kd": train_metrics["kd"],
                "val_acc": validation_accuracy,
                "val_f1": validation_f1,
            }
            history.append(epoch_record)

            print(
                f"[Fold {fold}] Epoch {epoch}: "
                f"loss={train_metrics['loss']:.4f} "
                f"ce={train_metrics['ce']:.4f} "
                f"kd={train_metrics['kd']:.4f} "
                f"val_acc={validation_accuracy:.4f} "
                f"val_f1={validation_f1:.4f}"
            )

            # Preserve the original protocol: select only by validation accuracy.
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_f1 = validation_f1
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "epoch": epoch,
                        "val_acc": validation_accuracy,
                        "val_f1": validation_f1,
                        "fold": fold,
                        "method": "LS-KD",
                        "hyperparameters": hyperparameters,
                    },
                    best_checkpoint_path,
                )

            with open(history_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_validation_accuracy,
                        "best_val_f1": best_validation_f1,
                        "history": history,
                    },
                    file,
                    indent=2,
                )

        if not os.path.isfile(best_checkpoint_path):
            raise RuntimeError(
                f"No best checkpoint was saved for fold {fold}."
            )
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location=device
        )
        student.load_state_dict(best_checkpoint["model_state_dict"])
        test_accuracy, test_f1 = trainer.evaluate(test_loader)

        fold_result = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_acc": best_validation_accuracy,
            "best_val_f1": best_validation_f1,
            "test_acc": test_accuracy,
            "test_f1": test_f1,
        }
        fold_results.append(fold_result)
        print(
            f"[Fold {fold}] Test Acc={test_accuracy:.4f} "
            f"F1={test_f1:.4f}"
        )

        # Release the large teacher before constructing the next fold.
        del trainer, optimizer, student, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    test_accuracies = np.asarray(
        [result["test_acc"] for result in fold_results], dtype=np.float64
    )
    test_f1_scores = np.asarray(
        [result["test_f1"] for result in fold_results], dtype=np.float64
    )
    ddof = 1 if len(fold_results) > 1 else 0
    summary = {
        "method": "LS-KD",
        "hyperparameters": hyperparameters,
        "mean_test_acc": float(test_accuracies.mean()),
        "std_test_acc": float(test_accuracies.std(ddof=ddof)),
        "mean_test_f1": float(test_f1_scores.mean()),
        "std_test_f1": float(test_f1_scores.std(ddof=ddof)),
        "folds": fold_results,
    }

    print("\n=== FINAL LS-KD RESULT ===")
    print(
        f"Test Acc : {summary['mean_test_acc']:.4f} "
        f"+/- {summary['std_test_acc']:.4f}"
    )
    print(
        f"Test F1  : {summary['mean_test_f1']:.4f} "
        f"+/- {summary['std_test_f1']:.4f}"
    )
    with open(
        os.path.join(save_root, "kd_lskd_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="MM-Fi 5-fold cross-modal LS-KD baseline."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
    )
    parser.add_argument(
        "--data_root",
        default="data/MMFi",
    )
    parser.add_argument(
        "--teacher_root",
        default="checkpoints/rgb_baseline",
    )
    parser.add_argument(
        "--save_root",
        default="checkpoints/mmwave_distilled_lskd",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument(
        "--kd_weight",
        type=float,
        default=9.0,
        help=(
            "Official LS-KD recommended value. For an additional "
            "strict common-coefficient comparison with KD_Align.py, "
            "run explicitly with --kd_weight 0.5."
        ),
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    run_distillation(
        config_path=args.config,
        data_root=args.data_root,
        teacher_root=args.teacher_root,
        save_root=args.save_root,
        device_string=args.device,
        epochs_override=args.epochs,
        learning_rate=args.lr,
        ce_weight=args.ce_weight,
        kd_weight=args.kd_weight,
        temperature=args.temperature,
        seed=args.seed,
    )
