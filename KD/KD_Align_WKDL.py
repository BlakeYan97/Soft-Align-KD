"""
MM-Fi cross-modal knowledge distillation with WKD-L.

Only the distillation objective is changed relative to KD_Align.py:
  - Teacher: the existing camera-derived 2D-keypoint GrayTeacherNet checkpoint.
  - Student: the existing PointNetLSTM with K=4 segment pooling and reweighting.
  - Split, data loaders, Adam optimizer, validation selection, and 5-fold test
    protocol follow the original project.

WKD-L source:
  Lv et al., "Wasserstein Distance Rivals Kullback-Leibler Divergence for
  Knowledge Distillation", NeurIPS 2024.

The Sinkhorn transport and target/non-target separation below follow the
authors' official implementation. By default, a fold-specific linear-CKA cost
matrix is generated from the frozen teacher logits on that fold's training
subjects, following the official WKD preprocessing. The cheaper teacher
classifier-prototype option is also available through --cost_matrix_mode fc.
"""

import argparse
import json
import math
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
            "and the class cost matrix would therefore be invalid."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded teacher checkpoint: {len(matched)}/{len(model_state)} "
        f"model tensors matched; missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )


def build_fc_cost_matrix(
    teacher: GrayTeacherNet,
    sharpening: float = 1.0,
) -> torch.Tensor:
    """
    Build the official WKD "fc" class cost from teacher class prototypes.

    Initial cost: 1 - cosine_similarity(w_i, w_j).
    If sharpening != 0: cost <- 1 - exp(-sharpening * cost).
    """
    classifier = getattr(teacher, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise AttributeError(
            "GrayTeacherNet must expose classifier.weight for WKD-L's "
            "FC-prototype cost matrix."
        )

    with torch.no_grad():
        prototypes = classifier.weight.detach().float()
        normalized = F.normalize(prototypes, p=2, dim=-1)
        cost = 1.0 - normalized @ normalized.transpose(0, 1)
        cost = F.relu(cost)
        if sharpening != 0:
            cost = 1.0 - torch.exp(-sharpening * cost)
        cost = F.relu(cost)
        cost.fill_diagonal_(0.0)
    return cost


def linear_cka_precenter(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Official linear CKA computation for two class-logit matrices."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    cross = y.transpose(0, 1) @ x
    hsic = torch.norm(cross, p="fro") ** 2
    x_variance = torch.norm(x.transpose(0, 1) @ x, p="fro")
    y_variance = torch.norm(y.transpose(0, 1) @ y, p="fro")
    return hsic / (x_variance * y_variance).clamp_min(1e-12)


def build_cka_cost_matrix(
    teacher: GrayTeacherNet,
    loader,
    device,
    num_classes: int,
    sharpening: float = 1.0,
) -> torch.Tensor:
    """
    Generate the fold-specific WKD linear-CKA matrix from teacher logits.

    Only the current fold's training subjects are used. Each class is truncated
    to the minimum class count, exactly as in WKD's official logit-prototype
    generation script.
    """
    class_logits = [[] for _ in range(num_classes)]
    teacher.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader, desc="Generating WKD CKA cost", leave=False
        ):
            teacher_input = (
                batch["input_rgb"].unsqueeze(2).float().to(device)
            )
            labels = actions_to_labels(batch["action"])
            logits = teacher(teacher_input).float().cpu()
            for index, label in enumerate(labels.tolist()):
                class_logits[label].append(logits[index])

    class_counts = [len(items) for items in class_logits]
    if min(class_counts) == 0:
        raise RuntimeError(
            "Cannot generate WKD CKA cost: at least one action class is "
            f"absent from the fold's training data. Counts={class_counts}"
        )
    minimum_count = min(class_counts)
    class_features = torch.stack(
        [
            torch.stack(items[:minimum_count], dim=0)
            for items in class_logits
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    cost = torch.zeros(
        num_classes,
        num_classes,
        device=device,
        dtype=torch.float32,
    )
    for first in range(num_classes):
        for second in range(first, num_classes):
            similarity = linear_cka_precenter(
                class_features[first], class_features[second]
            )
            distance = 1.0 - similarity
            cost[first, second] = distance
            cost[second, first] = distance

    cost = F.relu(cost)
    if sharpening != 0:
        cost = 1.0 - torch.exp(-sharpening * cost)
    cost = F.relu(cost)
    cost.fill_diagonal_(0.0)
    return cost


def load_external_cost_matrix(
    path: str,
    num_classes: int,
    device,
    sharpening: float,
) -> torch.Tensor:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, Mapping):
        for key in ("cost_matrix", "dist", "distance"):
            if key in raw and torch.is_tensor(raw[key]):
                raw = raw[key]
                break
    if not torch.is_tensor(raw):
        raise TypeError("The external WKD cost matrix must be a tensor.")

    cost = raw.detach().to(device=device, dtype=torch.float32)
    expected_shape = (num_classes, num_classes)
    if tuple(cost.shape) != expected_shape:
        raise ValueError(
            f"Expected cost matrix shape {expected_shape}, got "
            f"{tuple(cost.shape)}."
        )
    cost = F.relu(cost)
    if sharpening != 0:
        cost = 1.0 - torch.exp(-sharpening * cost)
    cost = F.relu(cost)
    cost.fill_diagonal_(0.0)
    return cost


def sinkhorn_transport(
    student_probability: torch.Tensor,
    teacher_probability: torch.Tensor,
    cost_matrix: torch.Tensor,
    regularization: float = 0.05,
    iterations: int = 10,
) -> torch.Tensor:
    """Entropic Sinkhorn transport used in the official WKD implementation."""
    if regularization <= 0:
        raise ValueError("Sinkhorn regularization must be positive.")
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be at least 1.")

    batch_size, dimension = student_probability.shape
    student_probability = student_probability.unsqueeze(-1)
    teacher_probability = teacher_probability.unsqueeze(-1)

    u = torch.full_like(student_probability, 1.0 / dimension)
    kernel = torch.exp(-cost_matrix / regularization)
    kernel_transpose = kernel.transpose(2, 1)

    for _ in range(iterations):
        v = teacher_probability / (
            torch.bmm(kernel_transpose, u) + 1e-8
        )
        u = student_probability / (torch.bmm(kernel, v) + 1e-8)

    return (
        u.reshape(batch_size, -1, 1)
        * kernel
        * v.reshape(batch_size, 1, -1)
    )


def wasserstein_non_target_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    cost_matrix: torch.Tensor,
    sinkhorn_regularization: float,
    sinkhorn_iterations: int,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    student_probability = F.softmax(
        student_logits / temperature, dim=-1
    ).float()
    teacher_probability = F.softmax(
        teacher_logits / temperature, dim=-1
    ).float()
    cost_matrix = F.relu(cost_matrix).float() + 1e-8

    flow = sinkhorn_transport(
        student_probability,
        teacher_probability,
        cost_matrix,
        regularization=sinkhorn_regularization,
        iterations=sinkhorn_iterations,
    )
    return (flow * cost_matrix).sum(dim=-1).sum(dim=-1).mean()


def wkd_l_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    gamma: float,
    full_cost_matrix: torch.Tensor,
    sinkhorn_regularization: float = 0.05,
    sinkhorn_iterations: int = 10,
):
    """
    Official WKD-L target/non-target separation.

    The ground-truth category uses teacher-weighted cross-entropy. The other
    categories use a discrete Wasserstein distance with the class cost matrix.
    """
    labels = labels.view(-1, 1)
    batch_size, num_classes = student_logits.shape

    student_log_probability = F.log_softmax(student_logits, dim=1)
    teacher_probability = F.softmax(teacher_logits, dim=1)
    target_student_log_probability = torch.gather(
        student_log_probability, 1, labels
    )
    target_teacher_probability = torch.gather(
        teacher_probability, 1, labels
    ).detach()
    target_loss = -(
        target_teacher_probability * target_student_log_probability
    ).mean()

    non_target_mask = torch.ones_like(student_logits).scatter_(
        1, labels, 0
    ).bool()
    student_non_target_logits = student_logits[non_target_mask].reshape(
        batch_size, num_classes - 1
    )
    teacher_non_target_logits = teacher_logits[non_target_mask].reshape(
        batch_size, num_classes - 1
    )

    batch_cost_matrix = full_cost_matrix.unsqueeze(0).expand(
        batch_size, -1, -1
    )
    pair_mask = non_target_mask.unsqueeze(1) & non_target_mask.unsqueeze(2)
    non_target_cost_matrix = batch_cost_matrix[pair_mask].reshape(
        batch_size, num_classes - 1, num_classes - 1
    )

    wasserstein_loss = wasserstein_non_target_loss(
        student_non_target_logits,
        teacher_non_target_logits,
        temperature=temperature,
        cost_matrix=non_target_cost_matrix,
        sinkhorn_regularization=sinkhorn_regularization,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    total = target_loss + gamma * wasserstein_loss
    return total, target_loss, wasserstein_loss


def scheduled_gamma(
    base_gamma: float,
    epoch: int,
    total_epochs: int,
    cosine_decay_start: int,
) -> float:
    """
    Match WKD's optional cosine decay.

    Set cosine_decay_start to a negative value to keep gamma fixed.
    """
    if cosine_decay_start < 0 or epoch <= cosine_decay_start:
        return base_gamma
    denominator = max(total_epochs - cosine_decay_start, 1)
    progress = (epoch - cosine_decay_start) / denominator
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_gamma * (1.0 + math.cos(progress * math.pi))


class WKDLTrainer:
    def __init__(
        self,
        teacher,
        student,
        optimizer,
        device,
        cost_matrix,
        ce_weight: float,
        base_gamma: float,
        temperature: float,
        sinkhorn_regularization: float,
        sinkhorn_iterations: int,
        total_epochs: int,
        cosine_decay_start: int,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.optimizer = optimizer
        self.device = device
        self.cost_matrix = cost_matrix.to(device=device, dtype=torch.float32)
        self.ce_weight = ce_weight
        self.base_gamma = base_gamma
        self.temperature = temperature
        self.sinkhorn_regularization = sinkhorn_regularization
        self.sinkhorn_iterations = sinkhorn_iterations
        self.total_epochs = total_epochs
        self.cosine_decay_start = cosine_decay_start

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train_one_epoch(self, loader, epoch: int):
        self.student.train()
        gamma = scheduled_gamma(
            self.base_gamma,
            epoch,
            self.total_epochs,
            self.cosine_decay_start,
        )
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "wkd": 0.0,
            "target": 0.0,
            "wasserstein": 0.0,
        }
        sample_count = 0

        for batch in tqdm(loader, desc="WKD-L training", leave=False):
            mmwave = preprocess_mmwave(batch, self.device)
            teacher_input = (
                batch["input_rgb"].unsqueeze(2).float().to(self.device)
            )
            labels = actions_to_labels(batch["action"]).to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(teacher_input).float()
            student_logits = self.student(mmwave).float()

            ce_loss = F.cross_entropy(student_logits, labels)
            wkd_loss, target_loss, wasserstein_loss = wkd_l_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=self.temperature,
                gamma=gamma,
                full_cost_matrix=self.cost_matrix,
                sinkhorn_regularization=self.sinkhorn_regularization,
                sinkhorn_iterations=self.sinkhorn_iterations,
            )
            loss = self.ce_weight * ce_loss + wkd_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite WKD-L loss encountered. Inspect the cost "
                    "matrix, logits, and Sinkhorn hyperparameters."
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["ce"] += ce_loss.item() * batch_size
            totals["wkd"] += wkd_loss.item() * batch_size
            totals["target"] += target_loss.item() * batch_size
            totals["wasserstein"] += wasserstein_loss.item() * batch_size

        if sample_count == 0:
            raise RuntimeError("The training loader produced no samples.")
        metrics = {
            key: value / sample_count for key, value in totals.items()
        }
        metrics["gamma"] = gamma
        return metrics

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
    gamma: float = 30.0,
    temperature: float = 2.0,
    sinkhorn_regularization: float = 0.05,
    sinkhorn_iterations: int = 10,
    cost_matrix_sharpening: float = 1.0,
    cost_matrix_mode: str = "cka",
    cost_matrix_path: Optional[str] = None,
    cosine_decay_start: int = 0,
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

    if cost_matrix_path:
        cost_source = cost_matrix_path
    elif cost_matrix_mode == "cka":
        cost_source = "fold_training_teacher_logits_linear_cka"
    elif cost_matrix_mode == "fc":
        cost_source = "teacher_classifier_weights"
    else:
        raise ValueError(
            "cost_matrix_mode must be either 'cka' or 'fc'."
        )
    hyperparameters: Dict[str, object] = {
        "method": "WKD-L",
        "ce_weight": ce_weight,
        "base_gamma": gamma,
        "temperature": temperature,
        "sinkhorn_regularization": sinkhorn_regularization,
        "sinkhorn_iterations": sinkhorn_iterations,
        "cost_matrix_sharpening": cost_matrix_sharpening,
        "cost_matrix_mode": cost_matrix_mode,
        "cost_matrix_source": cost_source,
        "cosine_decay_start": cosine_decay_start,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "epochs": epochs,
        "train_batch_size": 32,
        "eval_batch_size": 16,
        "seed": seed,
    }
    print("WKD-L settings:", json.dumps(hyperparameters, indent=2))

    fold_results = []
    for fold in range(num_folds):
        print("\n===============================")
        print(f"WKD-L TRAINING - Fold {fold}")
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
        cost_loader = make_dataloader(
            train_dataset, False, generator=generator, batch_size=16
        )

        teacher = GrayTeacherNet(num_classes=len(ACTIONS)).to(device)
        teacher_checkpoint = os.path.join(
            teacher_root,
            f"fold_{fold}",
            "best_res18_pretrain.pth",
        )
        load_teacher_checkpoint(teacher, teacher_checkpoint, device)

        if cost_matrix_path:
            resolved_cost_matrix_path = cost_matrix_path.format(fold=fold)
            cost_matrix = load_external_cost_matrix(
                resolved_cost_matrix_path,
                num_classes=len(ACTIONS),
                device=device,
                sharpening=cost_matrix_sharpening,
            )
            current_cost_source = os.path.abspath(resolved_cost_matrix_path)
        elif cost_matrix_mode == "cka":
            cost_matrix = build_cka_cost_matrix(
                teacher,
                cost_loader,
                device=device,
                num_classes=len(ACTIONS),
                sharpening=cost_matrix_sharpening,
            )
            current_cost_source = cost_source
        else:
            cost_matrix = build_fc_cost_matrix(
                teacher, sharpening=cost_matrix_sharpening
            ).to(device)
            current_cost_source = cost_source

        if not torch.isfinite(cost_matrix).all():
            raise FloatingPointError(
                f"Fold {fold} cost matrix contains non-finite values."
            )
        print(
            f"Cost matrix: source={current_cost_source}, "
            f"min={cost_matrix.min().item():.6f}, "
            f"max={cost_matrix.max().item():.6f}, "
            f"mean={cost_matrix.mean().item():.6f}"
        )

        student = PointNetLSTM(num_classes=len(ACTIONS)).to(device)
        optimizer = torch.optim.Adam(
            student.parameters(), lr=learning_rate
        )
        trainer = WKDLTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            device=device,
            cost_matrix=cost_matrix,
            ce_weight=ce_weight,
            base_gamma=gamma,
            temperature=temperature,
            sinkhorn_regularization=sinkhorn_regularization,
            sinkhorn_iterations=sinkhorn_iterations,
            total_epochs=epochs,
            cosine_decay_start=cosine_decay_start,
        )

        fold_directory = os.path.join(save_root, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)
        torch.save(
            cost_matrix.detach().cpu(),
            os.path.join(fold_directory, "wkd_cost_matrix.pth"),
        )
        best_checkpoint_path = os.path.join(
            fold_directory, "best_student_wkdl.pth"
        )
        history_path = os.path.join(fold_directory, "history_wkdl.json")

        best_validation_accuracy = -1.0
        best_validation_f1 = -1.0
        best_epoch = -1
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = trainer.train_one_epoch(train_loader, epoch)
            validation_accuracy, validation_f1 = trainer.evaluate(
                validation_loader
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce"],
                "train_wkd": train_metrics["wkd"],
                "train_target": train_metrics["target"],
                "train_wasserstein": train_metrics["wasserstein"],
                "gamma": train_metrics["gamma"],
                "val_acc": validation_accuracy,
                "val_f1": validation_f1,
            }
            history.append(epoch_record)

            print(
                f"[Fold {fold}] Epoch {epoch}: "
                f"loss={train_metrics['loss']:.4f} "
                f"ce={train_metrics['ce']:.4f} "
                f"wkd={train_metrics['wkd']:.4f} "
                f"target={train_metrics['target']:.4f} "
                f"wasserstein={train_metrics['wasserstein']:.6f} "
                f"gamma={train_metrics['gamma']:.4f} "
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
                        "method": "WKD-L",
                        "hyperparameters": hyperparameters,
                        "cost_matrix": cost_matrix.detach().cpu(),
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

        del trainer, optimizer, student, teacher, cost_matrix
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
        "method": "WKD-L",
        "hyperparameters": hyperparameters,
        "mean_test_acc": float(test_accuracies.mean()),
        "std_test_acc": float(test_accuracies.std(ddof=ddof)),
        "mean_test_f1": float(test_f1_scores.mean()),
        "std_test_f1": float(test_f1_scores.std(ddof=ddof)),
        "folds": fold_results,
    }

    print("\n=== FINAL WKD-L RESULT ===")
    print(
        f"Test Acc : {summary['mean_test_acc']:.4f} "
        f"+/- {summary['std_test_acc']:.4f}"
    )
    print(
        f"Test F1  : {summary['mean_test_f1']:.4f} "
        f"+/- {summary['std_test_f1']:.4f}"
    )
    with open(
        os.path.join(save_root, "kd_wkdl_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="MM-Fi 5-fold cross-modal WKD-L baseline."
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
        default="checkpoints/teacher",
    )
    parser.add_argument(
        "--save_root",
        default="outputs/wkdl",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=30.0,
        help="WKD Wasserstein weight; 30 is the official ImageNet setting.",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--sinkhorn_regularization", type=float, default=0.05
    )
    parser.add_argument("--sinkhorn_iterations", type=int, default=10)
    parser.add_argument("--cost_matrix_sharpening", type=float, default=1.0)
    parser.add_argument(
        "--cost_matrix_mode",
        choices=("cka", "fc"),
        default="cka",
        help=(
            "Default cka reproduces WKD's linear-CKA preprocessing from each "
            "fold's training teacher logits. fc uses teacher classifier "
            "weights and is faster."
        ),
    )
    parser.add_argument(
        "--cost_matrix_path",
        default=None,
        help=(
            "Optional 27x27 cost tensor; overrides --cost_matrix_mode. "
            "Use a {fold} placeholder for fold-specific files."
        ),
    )
    parser.add_argument(
        "--cosine_decay_start",
        type=int,
        default=0,
        help=(
            "Epoch at which gamma starts cosine decay. The official ImageNet "
            "configuration inherits 0; use -1 to disable decay."
        ),
    )
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
        gamma=args.gamma,
        temperature=args.temperature,
        sinkhorn_regularization=args.sinkhorn_regularization,
        sinkhorn_iterations=args.sinkhorn_iterations,
        cost_matrix_sharpening=args.cost_matrix_sharpening,
        cost_matrix_mode=args.cost_matrix_mode,
        cost_matrix_path=args.cost_matrix_path,
        cosine_decay_start=args.cosine_decay_start,
        seed=args.seed,
    )
