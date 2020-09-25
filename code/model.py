""" Deep neural network based on [1].

References:
    [1] -
"""

from torch import nn


class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat_block1 = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.feat_block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.feat_block3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.feat_block4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.feat_block5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.head_layers = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(4608, 1)
        )

    def forward(self, x):
        x = self.feat_block1(x)
        x = self.feat_block2(x)
        x = self.feat_block3(x)
        x = self.feat_block4(x)
        x = self.feat_block5(x)

        x = self.head_layers(x)
        return x