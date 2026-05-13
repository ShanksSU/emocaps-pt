import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(s, eps=1e-20):
    n = torch.norm(s, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (s / (n + eps))


def squash_hinton(s, eps=1e-20):
    n = torch.norm(s, dim=-1, keepdim=True)
    return (n ** 2 / (1 + n ** 2) / (n + eps)) * s

class PrimaryCaps(nn.Module):
    def __init__(self, F, K, N, D, s=1):
        super().__init__()
        self.D = D
        self.dw_conv = nn.Conv2d(F, F, K, stride=s, groups=F, padding=0)

    def forward(self, x):
        x = self.dw_conv(x)                        # [B, F, H', W']
        batch = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, H', W', F]
        x = x.view(batch, -1, self.D)              # [B, N, D]
        return squash(x)


class FCCaps(nn.Module):
    def __init__(self, N, D, input_N, input_D):
        super().__init__()
        self.D = D
        self.W = nn.Parameter(torch.empty(N, input_N, input_D, D))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(N, input_N, 1))

    def forward(self, inputs):
        u = torch.einsum('...ji,kjiz->...kjz', inputs, self.W)    # [B, N, input_N, D]
        c = torch.einsum('...ij,...kj->...i', u, u).unsqueeze(-1) # [B, N, input_N, 1]
        c = c / (self.D ** 0.5)
        c = F.softmax(c, dim=1)   # softmax over output capsule dim
        c = c + self.b
        s = (u * c).sum(dim=-2)   # [B, N, D]
        return squash(s)


class Length(nn.Module):
    def forward(self, x):
        return torch.sqrt((x ** 2).sum(dim=-1) + 1e-8)


class Mask(nn.Module):
    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs, mask = inputs
        else:
            lengths = torch.sqrt((inputs ** 2).sum(dim=-1))
            mask = F.one_hot(lengths.argmax(dim=-1),
                             num_classes=lengths.shape[-1]).float()
        return (inputs * mask.unsqueeze(-1)).view(inputs.shape[0], -1)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_shape):
        super().__init__()
        out_dim = int(np.prod(output_shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
            nn.Sigmoid(),
        )
        self.output_shape = tuple(output_shape)

    def forward(self, x):
        return self.net(x).view(x.shape[0], *self.output_shape)


class EfficientCapsNetDEAP(nn.Module):
    def __init__(self, input_shape, num_class, num_channels=32):
        super().__init__()
        H, W, C = input_shape   # 128, 32, 1

        # 4 conv layers, padding=0 (valid)
        self.conv1 = nn.Conv2d(C,            num_channels, kernel_size=5, padding=0)
        self.bn1   = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, 64,           kernel_size=3, padding=0)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,           64,           kernel_size=3, padding=0)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,           128,          kernel_size=3, stride=2, padding=0)
        self.bn4   = nn.BatchNorm2d(128)

        # Spatial size after each valid-padding layer
        H1 = H - 4;              W1 = W - 4              # 124, 28  (k=5)
        H2 = H1 - 2;             W2 = W1 - 2             # 122, 26  (k=3)
        H3 = H2 - 2;             W3 = W2 - 2             # 120, 24  (k=3)
        H4 = (H3 - 3) // 2 + 1; W4 = (W3 - 3) // 2 + 1 #  59, 11  (k=3, s=2)

        capsule_D = 8
        output_D  = 16
        primary_N = 128 // capsule_D   # 16  (1×1 after PrimaryCaps)

        self.primary_caps = PrimaryCaps(F=128, K=(H4, W4), N=primary_N, D=capsule_D)
        self.fc_caps      = FCCaps(N=num_class, D=output_D,
                                   input_N=primary_N, input_D=capsule_D)
        self.length       = Length()
        self.mask         = Mask()
        self.decoder      = Decoder(input_dim=num_class * output_D,
                                    output_shape=input_shape)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y=None):
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, C, H, W]
        x = self.bn1(F.relu(self.conv1(x)))       # Conv → ReLU → BN
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.primary_caps(x)                  # [B, 16, 8]
        x = self.fc_caps(x)                       # [B, num_class, 16]
        lengths = self.length(x)
        masked  = self.mask([x, y] if y is not None else x)
        return x, lengths, self.decoder(masked)


class EfficientCapsNetSEEDVIG(nn.Module):
    def __init__(self, input_shape, num_class, num_channels=64):
        super().__init__()
        H, W, C = input_shape   # e.g. 7, 5, 1

        # 2 conv layers, padding=0 (valid)
        self.conv1 = nn.Conv2d(C,            num_channels, kernel_size=3, padding=0)
        self.bn1   = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0)
        self.bn2   = nn.BatchNorm2d(num_channels)

        # Spatial size after each valid-padding layer (k=3)
        H2 = H - 2; W2 = W - 2   # e.g. 5, 3
        H3 = H2 - 2; W3 = W2 - 2  # e.g. 3, 1

        capsule_D = 8
        output_D  = 16
        primary_N = num_channels // capsule_D   # 8

        self.primary_caps = PrimaryCaps(F=num_channels, K=(H3, W3),
                                        N=primary_N, D=capsule_D)
        self.fc_caps      = FCCaps(N=num_class, D=output_D,
                                   input_N=primary_N, input_D=capsule_D)
        self.length       = Length()
        self.mask         = Mask()
        self.decoder      = Decoder(input_dim=num_class * output_D,
                                    output_shape=input_shape)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y=None):
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, C, H, W]
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.primary_caps(x)                  # [B, primary_N, 8]
        x = self.fc_caps(x)                       # [B, num_class, 16]
        lengths = self.length(x)
        masked  = self.mask([x, y] if y is not None else x)
        return x, lengths, self.decoder(masked)
