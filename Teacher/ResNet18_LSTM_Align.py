import torch
import torch.nn as nn
import torchvision.models as models

class GrayTeacherNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pretrained: bool = True,
        align_dim: int = 256   # ⭐ 新增
    ):
        super().__init__()

        backbone = models.resnet18(pretrained=pretrained)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:] = (
                    backbone.conv1.weight.mean(dim=1, keepdim=True)
                )

        backbone.fc = nn.Identity()
        self.cnn = backbone
        cnn_out_dim = 512

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.temporal_dim = lstm_hidden * (2 if bidirectional else 1)

        # ⭐ Teacher 对齐头
        self.align_proj = nn.Linear(self.temporal_dim, align_dim)

        self.classifier = nn.Linear(self.temporal_dim, num_classes)

    def forward(self, x, return_features=False):
        if x.dim() == 4:
            B, T, H, W = x.shape
            x = x.unsqueeze(2)
        else:
            B, T, C, H, W = x.shape

        x = x.view(B * T, 1, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)

        lstm_out, _ = self.lstm(feat)  # (B, T, D_t)

        seq_feat, _ = torch.max(lstm_out, dim=1)
        logits = self.classifier(seq_feat)

        # ⭐ 投影到对齐空间
        aligned_seq = self.align_proj(lstm_out)  # (B, T, D_align)

        if return_features:
            return logits, aligned_seq
        else:
            return logits

