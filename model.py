import torch
import torch.nn as nn

class Unet3D(nn.Module):
    def __init__(self, num_classes):
        super(Unet3D, self).__init__()
        # 下采样
        self.stage_1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.stage_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.stage_3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self.stage_4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        self.stage_5 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 1024, 3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
        )

        # 上采样
        self.upsample_4 = nn.ConvTranspose3d(1024, 512, 2, stride=2)
        self.upsample_3 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.upsample_2 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.upsample_1 = nn.ConvTranspose3d(128, 64, 2, stride=2)

        self.stage_up_4 = nn.Sequential(
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        self.stage_up_3 = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self.stage_up_2 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.stage_up_1 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.final = nn.Conv3d(64, num_classes, 1)

    def forward(self, x):
        # 下采样
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)
        x5 = self.stage_5(x4)

        # 上采样 + skip connection
        u4 = self.upsample_4(x5)
        u4 = self.stage_up_4(torch.cat([u4, x4], dim=1))

        u3 = self.upsample_3(u4)
        u3 = self.stage_up_3(torch.cat([u3, x3], dim=1))

        u2 = self.upsample_2(u3)
        u2 = self.stage_up_2(torch.cat([u2, x2], dim=1))

        u1 = self.upsample_1(u2)
        u1 = self.stage_up_1(torch.cat([u1, x1], dim=1))

        out = self.final(u1)
        return out

#测试
if __name__ == "__main__":
    model = Unet3D(num_classes=3)
    x = torch.rand((2, 1, 64, 256, 256))  # (B, C, D, H, W)
    y = model(x)
    print("Output shape:", y.shape)  # 应该是 (2, 3, 64, 256, 256)
