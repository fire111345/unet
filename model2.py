import torch
import torch.nn as nn

class Unet3D_light(nn.Module):
    def __init__(self, num_classes):
        super(Unet3D_light, self).__init__()
        # 下采样
        self.stage_1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.stage_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        # 上采样
        self.upsample_2 = nn.ConvTranspose3d(32, 16, 2, stride=2)

        self.stage_up_1 = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.final = nn.Conv3d(16, num_classes, 1)

    def forward(self, x):
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)

        u1 = self.upsample_2(x2)
        u1 = self.stage_up_1(torch.cat([u1, x1], dim=1))

        out = self.final(u1)
        return out

#测试
if __name__ == "__main__":
    model = Unet3D_light(num_classes=4).cuda()
    x = torch.rand((1, 1, 16, 64, 64)).cuda()  #测试 (B,C,D,H,W)
    y = model(x)
    print("Output shape:", y.shape)  # 应该是 (1, 3, 16, 64, 64)
