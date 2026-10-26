import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_point_mask(x, point_mask=None):
    """Return a boolean (..., N) mask for channel-first (..., C, N) points.

    Prefer an explicit mask derived before normalization/augmentation. For the
    legacy model(x) interface, only all-zero rows across ALL input attributes
    are treated as padding. A real all-zero detection needs an explicit mask.
    """
    expected_shape = x.shape[:-2] + x.shape[-1:]
    if point_mask is None:
        return x.ne(0).any(dim=-2)
    if point_mask.dtype != torch.bool:
        raise TypeError("point_mask must have dtype torch.bool")
    if tuple(point_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"Expected point_mask shape {tuple(expected_shape)}, "
            f"got {tuple(point_mask.shape)}"
        )
    return point_mask.to(device=x.device)


def masked_point_max(x, point_mask=None):
    """Max over N in (B,C,N), excluding invalid points; empty frames -> zero.

    This masks point-wise aggregation, not BatchNorm's training statistics.
    It adds no learned parameters and preserves legacy state_dict keys.
    """
    if point_mask is None:
        return x.max(dim=-1).values
    if x.ndim != 3 or tuple(point_mask.shape) != (x.shape[0], x.shape[-1]):
        raise ValueError("Expected features (B,C,N) and point_mask (B,N)")
    masked = x.masked_fill(~point_mask.unsqueeze(1), float("-inf"))
    pooled = masked.max(dim=-1).values
    return torch.where(point_mask.any(dim=-1, keepdim=True),
                       pooled, torch.zeros_like(pooled))

# ============================================================
#  STN modules: point-validity-aware pooling
# ============================================================

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, point_mask=None):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = masked_point_max(x, point_mask)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).repeat(B, 1)
        x = x + iden
        return x.view(-1, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k 
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, point_mask=None):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = masked_point_max(x, point_mask)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + iden.repeat(B, 1)
        return x.view(-1, self.k, self.k)

def segment_max_pooling(lstm_out, num_segments):
    """
    lstm_out: (B, T, H)
    return  : (B, K, H)
    """
    B, T, H = lstm_out.shape
    seg_len = T // num_segments

    seg_feats = []
    for k in range(num_segments):
        start = k * seg_len
        end = T if k == num_segments - 1 else (k + 1) * seg_len
        seg_feat, _ = lstm_out[:, start:end, :].max(dim=1)
        seg_feats.append(seg_feat)

    return torch.stack(seg_feats, dim=1)

# ============================================================
#  PointNet feature extractor: mask captured before learned transforms
# ============================================================

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()

        self.conv1 = nn.Conv1d(5, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, point_mask=None):
        point_mask = resolve_point_mask(x, point_mask)
        # Never reconstruct validity from transformed features: padding can
        # become nonzero through convolution biases or BatchNorm.
        x = x.masked_fill(~point_mask.unsqueeze(1), 0)
        xyz = x[:, :3, :]
        extra = x[:, 3:, :]

        trans = self.stn(xyz, point_mask=point_mask)
        xyz = torch.bmm(xyz.transpose(2, 1), trans).transpose(2, 1)
        x = torch.cat([xyz, extra], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x, point_mask=point_mask)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = masked_point_max(x, point_mask)   # (B, 1024)

        return x


# ============================================================
#  ⭐ NEW: Post-LSTM Reweighting (SAFE)
# ============================================================

class PostReweight(nn.Module):
    """
    Lightweight channel-wise reweighting.
    Initialized as identity → does NOT hurt baseline.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))  # ⭐ NEW

    def forward(self, x):
        return x * self.scale

# ============================================================
#  PointNet + LSTM Student (MINIMALLY MODIFIED)
# ============================================================

class PointNetLSTM(nn.Module):
    def __init__(self,
                 num_classes,
                 lstm_hidden=256,
                 lstm_layers=1,
                 bidirectional=True,
                 num_segments=4,
                 align_dim=1024):  # align维度往Teacher对齐
        super().__init__()

        self.pointnet = PointNetfeat()

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.lstm_dim = lstm_hidden * (2 if bidirectional else 1)
        self.num_segments = num_segments
        self.align_dim = align_dim

        # ⭐ post-max compensation
        self.post_reweight = PostReweight(self.lstm_dim)

        self.align_proj = nn.Linear(self.lstm_dim, align_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False, point_mask=None):
        """x: (B,T,5,N); optional boolean point_mask: (B,T,N).

        Existing positional return_features calls remain compatible.
        Point masks do not remove empty frames from the temporal LSTM.
        """
        point_mask = resolve_point_mask(x, point_mask)
        B, T, _, _ = x.shape
        feats = []

        for t in range(T):
            feats.append(self.pointnet(x[:, t], point_mask=point_mask[:, t]))

        feats = torch.stack(feats, dim=1)   # (B, T, 1024)
        lstm_out, _ = self.lstm(feats)      # (B, T, H)

        # Segment pooling
        seg_feats = segment_max_pooling(
            lstm_out, self.num_segments
        )                                   # (B, K, H)

        # Post reweight
        seg_feats = self.post_reweight(seg_feats)

        # Classification
        seq_feat, _ = seg_feats.max(dim=1)  # (B, H)
        logits = self.classifier(seq_feat)

        # ⭐ 对齐空间：每个 segment 一个 embedding
        align_feats = self.align_proj(seg_feats)  # (B, K, D_align)

        if return_features:
            return logits, align_feats
        else:
            return logits

