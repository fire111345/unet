import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from copy import deepcopy

from model2 import Unet3D_light
from dataload import Heart3DDataset
from loss_optimizer_lr import ComboLoss, get_optimizer_and_scheduler


def dice_score(pred, target, num_classes):
  #  计算每类 Dice score
# pred: (B,C,D,H,W) logits
# target: (B,D,H,W)

    pred = torch.argmax(pred, dim=1)
    dice_per_class = []
    for c in range(1, num_classes):  # 不算背景
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2*inter + 1e-5)/(union + 1e-5)
        dice_per_class.append(dice.item())
    return np.mean(dice_per_class)

def train_model(model, train_dataset, val_dataset, device, num_classes=3,
                batch_size=1, epochs=50, initial_lr=0.01, max_iters=1000,
                checkpoint_dir="./checkpoints"):

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    criterion = ComboLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, initial_lr=initial_lr, max_iters=max_iters)

    best_val_dice = 0.0
    global_step = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")

        #验证
        model.eval()
        val_dice = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                val_dice += dice_score(outputs, lbls, num_classes)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        #保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            # 直接保存当前模型的 state_dict
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"✅ Best model updated at epoch {epoch}, Dice: {best_val_dice:.4f}")

            print("Training Finished.")
            print(f"Best Validation Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_root = "C:/Users/F/Desktop/3DU-net/processed_data"

    # 读取训练集 ID
    train_cases = [f.replace("_0000.nii.gz","") 
                   for f in os.listdir(os.path.join(processed_root, "imagesTr")) 
                   if f.endswith("_0000.nii.gz")]

    # 读取验证集 ID
    val_cases = [f.replace("_0000.nii.gz","") 
                 for f in os.listdir(os.path.join(processed_root, "imagesTs")) 
                 if f.endswith("_0000.nii.gz")]
    

    print(f"训练集数量: {len(train_cases)}, 验证集数量: {len(val_cases)}")
    train_dataset = Heart3DDataset(root_dir="C:/Users/F/Desktop/3DU-net/processed_data/",
                                        case_list=train_cases,
                                        data_type="Tr",
                                        patch_size=(64,128,128),
                                        augment=True)

    val_dataset = Heart3DDataset(root_dir="C:/Users/F/Desktop/3DU-net/processed_data/",
                                        case_list=val_cases,
                                        data_type="Ts",
                                        patch_size=(64,128,128),
                                        augment=False)

    model = Unet3D_light(num_classes=4).to(device)
    train_model(model, train_dataset, val_dataset, device, num_classes=3,
                batch_size=2, epochs=50, initial_lr=0.01, max_iters=1000)
