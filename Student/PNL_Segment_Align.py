import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class STN3d(nn.Module):  # 只对xyz做变换，学习一个坐标变换，使点云在进入特征提取前处于“规范姿态”，减少旋转、平移、点云采集视角差异
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

    def forward(self, x):
        # x: (B, 3, N)
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):  # 对中间特征空间再做一次对齐，提高几何一致性与判别能力
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

    def forward(self, x):
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        iden = iden.repeat(batchsize, 1)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

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

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x: (B, 5, N)
        xyz = x[:, :3, :]  # x, y, z
        extra_feat = x[:, 3:, :]   # d, I

        trans = self.stn(xyz)  # (B, 3, 3)
        xyz = xyz.transpose(2, 1)  # (B, N, 3)
        xyz = torch.bmm(xyz, trans)  # (B, N, 3)
        xyz = xyz.transpose(2, 1)  # (B, 3, N)

        x = torch.cat([xyz, extra_feat], dim=1)  # (B, 5, N)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]  # (B, 1024)

        return x, trans_feat

num_segments = 4  # 经验值：3~6 都可以
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

    return torch.stack(seg_feats, dim=1)  # (B, K, H)

class PointNetLSTM(nn.Module):
    def __init__(self,
                 num_classes,
                 lstm_hidden=256,
                 lstm_layers=1,
                 bidirectional=True,
                 num_segments=4,
                 align_dim=256   # ⭐ 新增
                 ):
        super(PointNetLSTM, self).__init__()

        self.num_segments = num_segments
        self.align_dim = align_dim

        self.pointnet = PointNetfeat(
            global_feat=True,
            feature_transform=True
        )

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.lstm_dim = lstm_hidden * (2 if bidirectional else 1)

        # ⭐ 关键：Student 对齐头
        self.align_proj = nn.Linear(self.lstm_dim, align_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        x: (B, T, 5, N)

        return:
            logits
            aligned_seg_feats: (B, K, D_align)
        """
        B, T, _, N = x.size()
        frame_feats = []

        for t in range(T):
            feat, _ = self.pointnet(x[:, t])   # (B, 1024)
            frame_feats.append(feat)

        frame_feats = torch.stack(frame_feats, dim=1)  # (B, T, 1024)

        lstm_out, _ = self.lstm(frame_feats)  # (B, T, H)

        seg_feats = segment_max_pooling(
            lstm_out,
            num_segments=self.num_segments
        )  # (B, K, H)

        seq_feat, _ = seg_feats.max(dim=1)  # (B, H)
        logits = self.classifier(seq_feat)

        # ⭐ 投影到对齐空间
        aligned_seg_feats = self.align_proj(seg_feats)  # (B, K, D_align)

        if return_features:
            return logits, aligned_seg_feats
        else:
            return logits

############################################################################################
"训练 / 验证 / sanity check：logits = model(x)"
"跨模态蒸馏时：logits, frame_feats, seq_feat = model(x, return_features=True)"
############################################################################################

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("PointNet + LSTM (MM-Fi, 5D point cloud) sanity check")

    # -------- basic config --------
    B = 2          # batch size
    T = 8          # sequence length
    N = 112        # number of points per frame (after padding)
    D = 5          # (x, y, z, d, I)
    num_classes = 27

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- dummy input --------
    x = torch.randn(B, T, D, N).to(device)

    # -------- model --------
    model = PointNetLSTM(
        num_classes=num_classes,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=True
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(logits.shape))
    print("Expected    :", (B, num_classes))
    print("Sanity check passed ✅")
    print("=" * 60)