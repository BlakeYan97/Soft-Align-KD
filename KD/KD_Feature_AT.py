import os
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

from mmfi import make_dataset, make_dataloader
from PNL_reweight_Segment_Align import PointNetLSTM
# from ResNet18_LSTM_Align import GrayTeacherNet  # 2026-01-30 不使用align头版本，因为要和学生对齐，但是我们预训练的Teacher是512维的
from ResNet18_LSTM import GrayTeacherNet # 2026-01-30 使用没有align头的预训练模型，修改Student的align头维度为512，Student往Teacher对齐

# ============================================================
#  Action labels
# ============================================================
ACTIONS = ['A%02d' % i for i in range(1, 28)]
action_to_idx = {a: i for i, a in enumerate(ACTIONS)}

def actions_to_labels(action_list):
    return torch.tensor([action_to_idx[a] for a in action_list],
                        dtype=torch.long)

def temporal_segment_pooling(t_attn, num_segments):
    """
    t_attn: (B, T)
    return: (B, K)
    """
    B, T = t_attn.shape
    seg_len = T // num_segments

    segs = []
    for k in range(num_segments):
        start = k * seg_len
        end = T if k == num_segments - 1 else (k + 1) * seg_len

        seg = t_attn[:, start:end].mean(dim=1)  # ⭐ mean 比 max 更稳定
        segs.append(seg)

    return torch.stack(segs, dim=1)  # (B, K)

def feature_kd_loss(student_feat, teacher_feat):
    """
    student_feat: (B, D)
    teacher_feat: (B, D)
    """
    s = F.normalize(student_feat, dim=-1)
    t = F.normalize(teacher_feat, dim=-1)

    return F.mse_loss(s, t)

def attention_kd_loss(s_seg_feats, t_seq_feats, num_segments):
    """
    s_seg_feats: (B, K, D)
    t_seq_feats: (B, T, D)
    """

    # ===== Student attention =====
    s_attn = s_seg_feats.pow(2).mean(dim=-1)   # (B, K)

    # ===== Teacher attention =====
    t_attn = t_seq_feats.pow(2).mean(dim=-1)   # (B, T)

    # ===== 对齐到 K 段 =====
    t_attn_seg = temporal_segment_pooling(t_attn, num_segments)  # (B, K)

    # ===== normalize（很重要！）=====
    s_attn = F.normalize(s_attn, dim=-1)
    t_attn_seg = F.normalize(t_attn_seg, dim=-1)

    return F.mse_loss(s_attn, t_attn_seg)

# ============================================================
#  mmWave preprocessing
# ============================================================
def preprocess_mmwave(batch, device):
    """
    input_mmwave: (B, T, N, D)
    return       : (B, T, D, N)
    """
    x = batch.get("input_mmwave", None)
    if x is None:
        raise RuntimeError("Batch missing input_mmwave")

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.to(device=device, dtype=torch.float32)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x

# ============================================================
#  Soft Temporal Alignment KD Loss
# ============================================================
def soft_alignment_kd_loss(student_seg, teacher_seq, tau=0.08):  # 2026-02-06 tau修改为0.08
    """
    student_seg : (B, K, D)   Student segment features
    teacher_seq : (B, T, D)   Teacher full temporal features

    实现公式：
    α_{k,t} = softmax( (s_k · t_t) / τ )
    t̃_k = Σ_t α_{k,t} * t_t
    L = || s_k - t̃_k ||^2
    """

    # (B, K, T)
    sim = torch.einsum('bkd,btd->bkt', student_seg, teacher_seq) / tau

    # 注意力权重 α
    attn = torch.softmax(sim, dim=-1)

    # Teacher 加权对齐
    t_align = torch.einsum('bkt,btd->bkd', attn, teacher_seq)

    # L2 normalize
    s_norm = F.normalize(student_seg, dim=-1)
    t_norm = F.normalize(t_align, dim=-1)

    loss = F.mse_loss(s_norm, t_norm)

    return loss, attn


# ============================================================
#  Trainer
# ============================================================
class DistillationTrainer:

    def __init__(self, teacher, student, optimizer, device,
                 attn_kd_weight=0.5):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.optimizer = optimizer
        self.device = device
        self.attn_kd_weight = attn_kd_weight
        self.teacher.eval()

    # --------------------------------------------------------
    def train_one_epoch(self, loader):
        self.student.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):

            mm = preprocess_mmwave(batch, self.device)
            rgb = batch["input_rgb"].unsqueeze(2).float().to(self.device)
            labels = actions_to_labels(batch["action"]).to(self.device)

            # ===== Teacher =====
            with torch.no_grad():
                t_logits, t_seq = self.teacher(rgb, return_features=True)  # (B, T, D)

            # ===== Student =====
            s_logits, s_seg = self.student(mm, return_features=True)  # (B, K, D)

            # ===== CE =====
            ce_loss = F.cross_entropy(s_logits, labels)

            # ===== Feature KD =====
            t_feat, _ = torch.max(t_seq, dim=1)   # (B, D)
            s_feat, _ = torch.max(s_seg, dim=1)   # (B, D)

            feat_loss = feature_kd_loss(s_feat, t_feat)

            # ===== Attention KD =====
            attn_loss = attention_kd_loss(
                s_seg,
                t_seq,
                num_segments=s_seg.shape[1]
            )

            # ===== FINAL =====
            loss = (
                ce_loss
                + 0.5 * feat_loss
                + 0.5 * attn_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # --------------------------------------------------------
    def evaluate(self, loader):
        self.student.eval()

        preds, gts = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):

                mm = preprocess_mmwave(batch, self.device)
                labels = actions_to_labels(batch["action"]).to(self.device)

                logits = self.student(mm)
                pred = torch.argmax(logits, dim=1)

                preds.extend(pred.cpu().tolist())
                gts.extend(labels.cpu().tolist())

        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="macro")
        return acc, f1


# ============================================================
#  Main KD pipeline (K-Fold)
# ============================================================
def run_distillation(config_path, data_root, teacher_root, save_root):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_folds = config["kfold"]["num_folds"]
    epochs = config.get("epochs", 50)

    os.makedirs(save_root, exist_ok=True)
    fold_results = []

    for fold in range(num_folds):

        print("\n===============================")
        print(f"KD TRAINING - Fold {fold}")
        print("===============================")

        config["kfold"]["current_fold"] = fold
        g = torch.Generator()
        train_ds, val_ds, test_ds = make_dataset(data_root, config)
        train_loader = make_dataloader(train_ds, True, generator=g, batch_size=32)
        val_loader   = make_dataloader(val_ds, False, generator=g, batch_size=16)
        test_loader  = make_dataloader(test_ds, False, generator=g, batch_size=16)

        # ---------- Teacher ----------
        teacher = GrayTeacherNet(num_classes=27).to(device)
        ckpt = os.path.join(teacher_root, f"fold_{fold}", "best_res18_pretrain.pth")
        teacher.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

        # ---------- Student ----------
        student = PointNetLSTM(num_classes=27).to(device)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            device=device,
            attn_kd_weight=0.5,
        )

        best_acc = 0.0
        fold_dir = os.path.join(save_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # ---------- Train ----------
        for epoch in range(1, epochs + 1):

            loss = trainer.train_one_epoch(train_loader)
            acc, f1 = trainer.evaluate(val_loader)

            print(f"[Fold {fold}] Epoch {epoch}: "
                  f"Loss={loss:.4f} Acc={acc:.4f} F1={f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {"model_state_dict": student.state_dict()},
                    os.path.join(fold_dir, "best_student_feature_at.pth")
                )

        # ---------- Test ----------
        ckpt = torch.load(os.path.join(fold_dir, "best_student_feature_at.pth"),
                          map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])

        test_acc, test_f1 = trainer.evaluate(test_loader)

        fold_results.append({
            "fold": fold,
            "test_acc": test_acc,
            "test_f1": test_f1
        })

        print(f"[Fold {fold}] Test Acc={test_acc:.4f} F1={test_f1:.4f}")

    # ---------- Summary ----------
    test_accs = np.array([r["test_acc"] for r in fold_results])
    test_f1s  = np.array([r["test_f1"]  for r in fold_results])

    summary = {
        "mean_test_acc": float(test_accs.mean()),
        "std_test_acc":  float(test_accs.std(ddof=1)),
        "mean_test_f1":  float(test_f1s.mean()),
        "std_test_f1":   float(test_f1s.std(ddof=1)),
        "folds": fold_results
    }

    print("\n=== FINAL KD RESULT ===")
    print(
        f"Test Acc : {summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}\n"
        f"Test F1  : {summary['mean_test_f1']:.4f} ± {summary['std_test_f1']:.4f}"
    )

    with open(os.path.join(save_root, "kd_feature_at_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ============================================================
if __name__ == "__main__":
    run_distillation(
        config_path="config.yaml",
        data_root="data/MMFi",
        teacher_root="checkpoints/teacher",
        save_root="outputs/feature_attention_kd"
    )
