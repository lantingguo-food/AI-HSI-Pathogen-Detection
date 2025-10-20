import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features, n_classes, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(in_features, n_classes)
        )
    def forward(self, x):
        return self.net(x)
