import torch
import torch.nn as nn
import torchvision.models as models


class GrayTeacherNet(nn.Module):
    """
    Teacher network for MM-Fi KD

    Input  : Grayscale video (B, T, 1, H, W)
    CNN    : ResNet-18 (shallow, texture-limited)
    Temporal: 2-layer Bi-LSTM (strong temporal modeling)
    Agg    : Temporal max pooling
    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        # -------------------------------------------------
        # 1. Frame-level CNN encoder (ResNet-18, grayscale)
        # -------------------------------------------------
        backbone = models.resnet18(pretrained=pretrained)

        # 🔴 修改第一层：3 → 1 channel
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # 用 RGB 权重均值初始化灰度卷积
            with torch.no_grad():
                backbone.conv1.weight[:] = (
                    backbone.conv1.weight.mean(dim=1, keepdim=True)
                )

        backbone.fc = nn.Identity()   # remove classifier
        self.cnn = backbone           # output: (B*T, 512)

        cnn_out_dim = 512

        # -------------------------------------------------
        # 2. Temporal encoder (Bi-LSTM)
        # -------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        temporal_dim = lstm_hidden * (2 if bidirectional else 1)

        # -------------------------------------------------
        # 3. Classification head
        # -------------------------------------------------
        self.classifier = nn.Linear(temporal_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, T, H, W) or (B, T, 1, H, W)
        """

        # -------- shape normalization --------
        if x.dim() == 4:
            B, T, H, W = x.shape
            x = x.unsqueeze(2)          # -> (B, T, 1, H, W)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            assert C == 1, "GrayTeacherNet expects 1-channel input"
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # -------- frame-level CNN --------
        x = x.view(B * T, 1, H, W)
        feat = self.cnn(x)              # (B*T, 512)
        feat = feat.view(B, T, -1)      # (B, T, 512)

        # -------- temporal modeling --------
        lstm_out, _ = self.lstm(feat)   # (B, T, D)

        # -------- temporal max pooling --------
        seq_feat, _ = torch.max(lstm_out, dim=1)  # (B, D)

        logits = self.classifier(seq_feat)

        if return_features:
            # return logits, seq_feat  # max pooling 后的特征
            return logits, lstm_out  # 输出 LSTM 输出以供 KD Segment 使用 2025-12-31
        else:
            return logits

        # if return_features:
        #     return logits, seq_feat  # max pooling 后的特征 # 2026-03-18 返回global feature以使用Feature KD，仅实验
        # else:
        #     return logits 
