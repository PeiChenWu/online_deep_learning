import torch
import torch.nn as nn
from pathlib import Path

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(logits, target)

class LinearClassifier(nn.Module):
    def __init__(self, h: int = 64, w: int = 64, num_classes: int = 6):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * h * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)

class Classifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(logits, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class Detector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.down1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(32, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = ConvBlock(32, 16)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))
        x = self.up1(x3)
        x = self.dec1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x1], dim=1))
        return self.seg_head(x), self.depth_head(x).squeeze(1)
        
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            seg_logits, depth_pred = self.forward(x)
            seg_preds = torch.argmax(seg_logits, dim=1)
            return seg_preds, depth_pred



model_factory = {
    "classifier": Classifier,
    "detector": Detector,
    "linear": LinearClassifier
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
