import torch
import torch.nn as nn

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )
        else:
            self.downsample = None
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, layers=(2,2,2,2), base_width=32, num_classes=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_width), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(base_width, base_width, layers[0])
        self.layer2 = self._make_layer(base_width, base_width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width*2, base_width*4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width*4, base_width*8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base_width*8, num_classes)
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = [BasicBlock3D(inplanes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.drop(x)
        return self.fc(torch.flatten(x, 1))

def resnet3d_18(in_channels=1, base=32, num_classes=2, dropout=0.1):
    return ResNet3D(in_channels=in_channels, layers=(2,2,2,2), base_width=base, num_classes=num_classes, dropout=dropout)
