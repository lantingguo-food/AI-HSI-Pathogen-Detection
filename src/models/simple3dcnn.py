import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, base=32, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1), nn.BatchNorm3d(base), nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1), nn.BatchNorm3d(base), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(base, base*2, 3, padding=1), nn.BatchNorm3d(base*2), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(base*2, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
