import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.carbonformer import CarbonFormer_v1
from dataset import CarbonDataset, CarbonDataset_csv
from models.util import select_device, mix_patch
from tqdm import tqdm
from models.metrics import CarbonLoss
import wandb
import os
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train CarbonFormer')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--cls_lambda', type=float, default=0.0, help='classification loss weight')
    parser.add_argument('--reg_lambda', type=float, default=1.0, help='regression loss weight')
    parser.add_argument('--source_fp', type=str, default='AP25_Forest_IMAGE.csv', help='source dataset file')
    parser.add_argument('--target_fp', type=str, default='AP25_City_IMAGE.csv', help='target dataset file')
    parser.add_argument('--pretrain', type=str, default=None, help='path to pretrained model')
    return parser.parse_args()

def train(epoch, device, model, optimizer, loss, loader, domain="Forest"):
    model.train()
    train_stats = {
        'total_loss': 0.0,
        'total_cls_loss': 0.0,
        'total_reg_loss': 0.0,
        'total_acc_c': 0.0,
        'total_acc_r': 0.0,
        'total_miou': 0.0,
        'batches': 0
    }
    
    for x, sh, carbon, gt in tqdm(loader, desc=f"Training {domain} Epoch {epoch+1}"):
        optimizer.zero_grad()
        x, sh, carbon, gt = x.to(device), sh.to(device), carbon.to(device), gt.to(device)
        gt_pred, carbon_pred = model(x, sh)
        total_loss, cls_loss, reg_loss, acc_c, acc_r, miou = loss(gt_pred, gt.squeeze(1), carbon_pred, carbon)
        
        total_loss.backward()
        optimizer.step()
        
        train_stats['total_loss'] += total_loss.item()
        train_stats['total_cls_loss'] += cls_loss.item()
        train_stats['total_reg_loss'] += reg_loss.item()
        train_stats['total_acc_c'] += acc_c
        train_stats['total_acc_r'] += acc_r
        train_stats['total_miou'] += miou
        train_stats['batches'] += 1
        
    return {k: v / train_stats['batches'] for k, v in train_stats.items() if k != 'batches'}

def validate(model, device, loss_fn, val_loader, target_val_loader):
    model.eval()
    val_stats = {
        'total_loss': 0.0,
        'total_cls_loss': 0.0,
        'total_reg_loss': 0.0,
        'total_acc_c': 0.0,
        'total_acc_r': 0.0,
        'total_miou': 0.0,
        'batches': 0
    }
    
    with torch.no_grad():
        for loader in [val_loader, target_val_loader]:
            for x, sh, carbon, gt in tqdm(loader, desc="Validating"):
                x, sh, carbon, gt = x.to(device), sh.to(device), carbon.to(device), gt.to(device)
                gt_pred, carbon_pred = model(x, sh)
                total_loss, cls_loss, reg_loss, acc_c, acc_r, miou = loss_fn(gt_pred, gt.squeeze(1), carbon_pred, carbon)
                
                val_stats['total_loss'] += total_loss.item()
                val_stats['total_cls_loss'] += cls_loss.item()
                val_stats['total_reg_loss'] += reg_loss.item()
                val_stats['total_acc_c'] += acc_c
                val_stats['total_acc_r'] += acc_r
                val_stats['total_miou'] += miou
                val_stats['batches'] += 1

    return {k: v / val_stats['batches'] for k, v in val_stats.items() if k != 'batches'}

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    FOLDER_PATH = {
        'AP10_Forest_IMAGE.csv': 4,
        'AP25_Forest_IMAGE.csv': 4,   
        'AP10_City_IMAGE.csv': 4,
        'AP25_City_IMAGE.csv': 4,
        'SN10_Forest_IMAGE.csv': 4,
    }
    
    label_size = 256 // 4
    model_args = {
        'dims': (32, 64, 160, 256),
        'heads': (1, 2, 5, 8),
        'ff_expansion': (8, 8, 4, 4),
        'reduction_ratio': (8, 4, 2, 1),
        'num_layers': (2, 2, 6, 2),
        'decoder_dim': 256,
        'divisor': 4,
        'channels': 3,
        'num_classes': FOLDER_PATH[args.source_fp],
        'stage_kernel_stride_pad': [(4, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1)],
    }

    device = select_device()
    source_dataset_name = args.source_fp.split(".")[0]
    target_dataset_name = args.target_fp.split(".")[0]
    model_name = "CarbonFormer_v1"
    checkpoint_path = f"checkpoints/{model_name}/"
    name = f"{model_name}"+"B0"+source_dataset_name.replace("_IMAGE", "")+f"_{label_size}_seg_0"
    
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb.login()
    wandb.init(project="CCP", name=name, config=vars(args))
    wandb.config.update(model_args)

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    sh_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    label_transform = transforms.Compose([
        transforms.Resize((label_size, label_size)),
    ])

    train_dataset = CarbonDataset_csv(args.source_fp, image_transform, sh_transform, label_transform, mode="Train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = CarbonDataset_csv(args.source_fp, image_transform, sh_transform, label_transform, mode="Valid")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    target_dataset = CarbonDataset_csv(args.target_fp, image_transform, sh_transform, label_transform, mode="Train")
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    target_val_dataset = CarbonDataset_csv(args.target_fp, image_transform, sh_transform, label_transform, mode="Valid")
    target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
     
    model = CarbonFormer_v1(**model_args).to(device)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain, map_location=device), strict=False)
    
    loss_fn = CarbonLoss(num_classes=FOLDER_PATH[args.source_fp], cls_lambda=args.cls_lambda, reg_lambda=args.reg_lambda).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        forest_stats = train(epoch, device, model, optimizer, loss_fn, train_loader)
        city_stats = train(epoch, device, model, optimizer, loss_fn, target_loader, domain="City")
        
        train_stats = {k: (forest_stats[k] + city_stats[k]) / 2 for k in forest_stats}
        
        logger.info(f"Epoch {epoch+1}, Train Loss: {train_stats['total_loss']:.4f}, "
                    f"Train cls_loss: {train_stats['total_cls_loss']:.4f}, "
                    f"Train reg_loss: {train_stats['total_reg_loss']:.4f}, "
                    f"Train acc_c: {train_stats['total_acc_c']:.4f}, "
                    f"Train acc_r: {train_stats['total_acc_r']:.4f}, "
                    f"Train miou: {train_stats['total_miou']:.4f}")
        
        wandb.log({"Train": train_stats, "epoch": epoch+1})

        val_stats = validate(model, device, loss_fn, val_loader, target_val_loader)
        
        logger.info(f"Validation Loss: {val_stats['total_loss']:.4f}, "
                    f"Validation cls_loss: {val_stats['total_cls_loss']:.4f}, "
                    f"Validation reg_loss: {val_stats['total_reg_loss']:.4f}, "
                    f"Validation acc_c: {val_stats['total_acc_c']:.4f}, "
                    f"Validation acc_r: {val_stats['total_acc_r']:.4f}, "
                    f"Validation miou: {val_stats['total_miou']:.4f}")
        
        wandb.log({"Validation": val_stats, "epoch": epoch+1})

        scheduler.step(val_stats['total_loss'])

        if val_stats['total_loss'] < best_val_loss:
            best_val_loss = val_stats['total_loss']
            torch.save(model.state_dict(), f"{checkpoint_path}/{name}_best.pth")
            logger.info(f"New best model saved at epoch {epoch+1}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, f"{checkpoint_path}/{name}_checkpoint_{epoch+1}.pth")

        torch.save(model.state_dict(), f"{checkpoint_path}/{name}_last.pth")

    wandb.finish()
    
if __name__ == "__main__":
    main()