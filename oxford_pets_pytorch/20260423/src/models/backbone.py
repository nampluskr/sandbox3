import os
import torch
import torch.nn as nn
import torchvision.models as models


BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "wide_resnet50_2-32ee1156.pth",
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b5": "efficientnet_b5_lukemelas-1a07897c.pth",
    "vgg16": "vgg16-397923af.pth",
    "vgg19": "vgg19-dcbb9e9d.pth",
    "vgg16_bn": "vgg16_bn-6c64b313.pth",
    "vgg19_bn": "vgg19_bn-c79401a0.pth",
    "mobilenet_v2": "mobilenet_v2-7ebf99e0.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
}


class CNNModel(nn.Module):
    def __init__(self, output_dim=10, in_channels=3):
        super().__init__()
        self.backbone =  nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_input_dim = 256 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc(x)
        return x


def get_backbone_path(backbone: str):
    backbone_dir = os.getenv('BACKBONE_DIR')

    if backbone_dir is None:
        raise RuntimeError("[ERROR] BACKBONE_DIR not set in environment variables.")

    filename = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.pth")
    weights_path = os.path.join(backbone_dir, filename)

    if os.path.isfile(weights_path):
        print(f"> {backbone} weights is loaded from {os.path.basename(weights_path)}.")
    else:
        print(f"> {backbone} weights not found in {os.path.basename(weights_path)}.")
    return weights_path


def get_pretrained_model(backbone, output_dim):
    if backbone not in BACKBONE_WEIGHT_FILES:
        raise ValueError(f"Unsupported backbone: {backbone}. Must be one of {list(BACKBONE_WEIGHT_FILES.keys())}.")

    if "resnet" in backbone or "wide_resnet" in backbone:
        model_fn = getattr(models, backbone)
        model = model_fn(weights=None)
        weight_path = get_backbone_path(backbone)
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, output_dim)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        weight_path = get_backbone_path(backbone)
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, output_dim)

    elif backbone == "efficientnet_b5":
        model = models.efficientnet_b5(weights=None)
        weight_path = get_backbone_path(backbone)
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, output_dim)

    elif "vgg" in backbone:
        model_fn = getattr(models, backbone)
        model = model_fn(weights=None)
        weight_path = get_backbone_path(backbone)
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, output_dim)

    elif "mobilenet" in backbone:
        model_fn = getattr(models, backbone)
        model = model_fn(weights=None)
        weight_path = get_backbone_path(backbone)
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        if "v2" in backbone:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, output_dim)
        else:  # v3
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, output_dim)

    else:
        raise NotImplementedError(f"Backbone '{backbone}' is not implemented.")

    return model
