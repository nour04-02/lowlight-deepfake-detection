import torch
import torch.nn as nn
import torchvision.models as models

try:
    import timm
except ImportError:
    timm = None

# ===============================
# SE Block
# ===============================
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ===============================
# ResNet34 + SE
# ===============================
class ResNet34SEOnly(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se = SqueezeExcitation(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))

# ===============================
# LIDeepDet
# ===============================
class LIDeepDet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        self.enhance = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = torch.clamp(x + self.enhance(x), 0, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))

# ===============================
# EfficientNet-B0
# ===============================
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        self.m = backbone   # ⭐ هذا السطر المهم

    def forward(self, x):
        return self.m(x)

# ===============================
# Xception
# ===============================
class Xception(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        assert timm is not None, "Please install timm"

        self.backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x):
        return self.backbone(x)

# ===============================
# Model Factory
# ===============================
def create_model(name):
    name = name.lower()

    if name == "resnet34_se_only":
        return ResNet34SEOnly()
    elif name == "lideepdet":
        return LIDeepDet()
    elif name == "efficientnetb0":
        return EfficientNetB0()
    elif name == "xception":
        return Xception()
    else:
        raise ValueError(f"Unknown model: {name}")

# ===============================
# Available Models
# ===============================
AVAILABLE_MODELS = [
    "resnet34_se_only",
    "lideepdet",
    "efficientnetb0",
    "xception"
]
