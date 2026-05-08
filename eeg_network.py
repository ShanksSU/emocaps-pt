import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Capsule activations
def squash(s, eps=1e-20):
    n = torch.norm(s, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (s / (n + eps))


def squash_hinton(s, eps=1e-20):
    n = torch.norm(s, dim=-1, keepdim=True)
    return (n ** 2 / (1 + n ** 2) / (n + eps)) * s


# Capsule layers
class PrimaryCaps(nn.Module):
    def __init__(self, F, K, N, D, s=1):
        super().__init__()
        self.D = D
        self.dw_conv = nn.Conv2d(F, F, K, stride=s, groups=F, padding=0)

    def forward(self, x):
        x = self.dw_conv(x)                        # [batch, F, H', W']
        batch = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous()    # [batch, H', W', F]
        x = x.view(batch, -1, self.D)              # [batch, N, D]
        return squash(x)


class FCCaps(nn.Module):
    def __init__(self, N, D, input_N, input_D):
        super().__init__()
        self.D = D
        self.W = nn.Parameter(torch.empty(N, input_N, input_D, D))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(N, input_N, 1))

    def forward(self, inputs):
        # inputs: [batch, input_N, input_D]
        u = torch.einsum('...ji,kjiz->...kjz', inputs, self.W)   # [batch, N, input_N, D]
        c = torch.einsum('...ij,...kj->...i', u, u).unsqueeze(-1) # [batch, N, input_N, 1]
        c = c / (self.D ** 0.5)
        c = F.softmax(c, dim=1)
        c = c + self.b
        s = (u * c).sum(dim=-2)   # [batch, N, D]
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
            mask = F.one_hot(lengths.argmax(dim=-1), num_classes=lengths.shape[-1]).float()
        return (inputs * mask.unsqueeze(-1)).view(inputs.shape[0], -1)


# EEG model architectures
class EfficientCapsNetDEAP(nn.Module):
    def __init__(self, input_shape, num_class, num_channels=32):
        super().__init__()
        H, W, C = input_shape   # 128, 32, 1

        self.conv1 = nn.Conv2d(C, num_channels, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        H2, W2    = math.ceil(H / 2), math.ceil(W / 2)   # 64, 16
        H3, W3    = H2 - 9 + 1, W2 - 9 + 1               # 56, 8
        capsule_D = 8
        primary_N = H3 * W3 * (128 // capsule_D)          # 7168

        self.primary_caps = PrimaryCaps(F=128, K=9, N=primary_N, D=capsule_D)
        self.fc_caps      = FCCaps(N=num_class, D=capsule_D, input_N=primary_N, input_D=capsule_D)
        self.length       = Length()
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

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()   # [batch, C, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.primary_caps(x)                  # [batch, primary_N, 8]
        x = self.fc_caps(x)                       # [batch, num_class, 8]
        return x, self.length(x)                  # caps_vectors, caps_lengths


class EfficientCapsNetSEEDVIG(nn.Module):
    def __init__(self, input_shape, num_class, num_channels=64):
        super().__init__()
        H, W, C = input_shape   # e.g. 17, 5, 1

        self.conv1 = nn.Conv2d(C, num_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(num_channels)

        K, capsule_D = 3, 8
        H2, W2    = H - K + 1, W - K + 1                          # 15, 3 (for 17-ch input)
        primary_N = H2 * W2 * (num_channels // capsule_D)          # 360

        self.primary_caps = PrimaryCaps(F=num_channels, K=K, N=primary_N, D=capsule_D)
        self.fc_caps      = FCCaps(N=num_class, D=capsule_D, input_N=primary_N, input_D=capsule_D)
        self.length       = Length()
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

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()   # [batch, C, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.primary_caps(x)                  # [batch, primary_N, 8]
        x = self.fc_caps(x)                       # [batch, num_class, 8]
        return x, self.length(x)
