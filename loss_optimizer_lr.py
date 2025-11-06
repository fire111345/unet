import torch
import torch.nn as nn
import torch.optim as optim


#Dice Loss + CrossEntropy Loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        #pred: (B, C, D, H, W) logits or softmax
        #target: (B, D, H, W) long
        num_classes = pred.shape[1]
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)  # one-hot encoding

        dims = (0, 2, 3, 4)  # sum over batch + spatial dims
        intersection = torch.sum(pred_soft * target_one_hot, dims)
        cardinality = torch.sum(pred_soft + target_one_hot, dims)
        dice_loss = 1.0 - (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return dice_loss.mean()


class ComboLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_ce=1.0):
        super(ComboLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

#Optimizer +  LR

def get_optimizer_and_scheduler(model, initial_lr=0.01, max_iters=1000):
    #initial_lr: 初始学习率
    #max_iters: 总迭代次数，用于 Poly LR
    
    # SGD + Nesterov
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)

    # Poly LR scheduler: lr = lr0 * (1 - iter/max_iters)^0.9
    lambda_poly = lambda iter: max((1 - iter / max_iters),0) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

    return optimizer, scheduler
#testing
if __name__ == "__main__":
    from model2 import Unet3D  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet3D(num_classes=3).to(device)

    criterion = ComboLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, initial_lr=0.01, max_iters=1000)

    # 测试 forward/backward
    x = torch.rand((2,1,64,128,128)).to(device)
    y = torch.randint(0,3,(2,64,128,128)).to(device)  # target
    loss = criterion(model(x), y)
    loss.backward()

    print("gooood")
