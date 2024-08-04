import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.segformer_simple import Segformerwithcarbon
from dataset_segwithcarbon import CarbonDataset, CarbonDataset_csv
from model.util import select_device, mix_patch
from tqdm import tqdm
from model.metrics import CarbonLoss , CarbonLossWithRMSE
import wandb
import os
import argparse
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser(description='Train Segformer with Carbon')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--cls_lambda', type=float, default=1.0, help='classification loss weight')
    parser.add_argument('--reg_lambda', type=float, default=0.005, help='regression loss weight')
    parser.add_argument('--source_fp', type=str, default='AP25_Forest_IMAGE.csv', help='source dataset file')
    parser.add_argument('--target_fp', type=str, default='AP25_City_IMAGE.csv', help='target dataset file')
    parser.add_argument('--pretrain', type=str, default=None, help='path to pretrained model')
    return parser.parse_args()

def get_transforms(label_size):
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
    return image_transform, sh_transform, label_transform

def train_epoch(epoch, device, model, optimizer, loss_fn, loader, domain="Forest"):
    model.train()
    train_stats = {'total_loss': 0.0, 'total_cls_loss': 0.0, 'total_reg_loss': 0.0,
                   'total_acc_c': 0.0, 'total_acc_r': 0.0, 'total_miou': 0.0, 'total_rmse': 0.0, 'batches': 0}
    
    for x, carbon, gt in tqdm(loader, desc=f"Training {domain} Epoch {epoch+1}"):
        optimizer.zero_grad()
        x, carbon, gt = x.to(device), carbon.to(device), gt.to(device)
        gt_pred, carbon_pred = model(x)
        total_loss, cls_loss, reg_loss, acc_c, acc_r, miou, rmse = loss_fn(gt_pred, gt.squeeze(1), carbon_pred, carbon)
        
        total_loss.backward()
        optimizer.step()
        
        train_stats['total_loss'] += total_loss.item()
        train_stats['total_cls_loss'] += cls_loss.item()
        train_stats['total_reg_loss'] += reg_loss.item()
        train_stats['total_acc_c'] += acc_c
        train_stats['total_acc_r'] += acc_r
        train_stats['total_miou'] += miou
        train_stats['total_rmse'] += rmse
        train_stats['batches'] += 1

    return {k: v / train_stats['batches'] for k, v in train_stats.items() if k != 'batches'}

def validate(model, device, loss_fn, loader, domain="Forest"):
    model.eval()
    val_stats = {'total_loss': 0.0, 'total_cls_loss': 0.0, 'total_reg_loss': 0.0,
                 'total_acc_c': 0.0, 'total_acc_r': 0.0, 'total_miou': 0.0, 'total_rmse': 0.0, 'batches': 0}
    
    with torch.no_grad():
        for x, carbon, gt in tqdm(loader, desc=f"Validating {domain}"):
            x, carbon, gt = x.to(device), carbon.to(device), gt.to(device)
            gt_pred, carbon_pred = model(x)
            total_loss, cls_loss, reg_loss, acc_c, acc_r, miou, rmse = loss_fn(gt_pred, gt.squeeze(1), carbon_pred, carbon)
            
            val_stats['total_loss'] += total_loss.item()
            val_stats['total_cls_loss'] += cls_loss.item()
            val_stats['total_reg_loss'] += reg_loss.item()
            val_stats['total_acc_c'] += acc_c
            val_stats['total_acc_r'] += acc_r
            val_stats['total_miou'] += miou
            val_stats['total_rmse'] += rmse
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
        'num_layers': (2, 2, 2, 2),
        'decoder_dim': 256,
        'channels': 4,
        'num_classes': 4,
        'stage_kernel_stride_pad': [(7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1)],
    }

    device = select_device()
    source_dataset_name = args.source_fp.split(".")[0]
    target_dataset_name = args.target_fp.split(".")[0]
    model_name = "SegFomer"
    checkpoint_path = f"checkpoints/{model_name}/"
    name = f"{model_name}"+"B0"+source_dataset_name.replace("_IMAGE", "")+f"_{label_size}"
    
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb.login()
    wandb.init(project="CCP", name=name, config=vars(args))
    wandb.config.update(model_args)

    image_transform, sh_transform, label_transform = get_transforms(label_size)

    train_dataset = CarbonDataset_csv(args.source_fp, image_transform, sh_transform, label_transform, mode="Train", combine=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = CarbonDataset_csv(args.source_fp, image_transform, sh_transform, label_transform, mode="Valid",combine=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    target_dataset = CarbonDataset_csv(args.target_fp, image_transform, sh_transform, label_transform, mode="Train", combine=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    target_val_dataset = CarbonDataset_csv(args.target_fp, image_transform, sh_transform, label_transform, mode="Valid", combine=True)
    target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
     
    model = Segformerwithcarbon(**model_args).to(device)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain, map_location=device), strict=False)
    
    loss_fn = CarbonLossWithRMSE(num_classes=FOLDER_PATH[args.source_fp], cls_lambda=args.cls_lambda, reg_lambda=args.reg_lambda).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        forest_stats = train_epoch(epoch, device, model, optimizer, loss_fn, train_loader)
        city_stats = train_epoch(epoch, device, model, optimizer, loss_fn, target_loader, domain="City")
        
        train_stats = {k: (forest_stats[k] + city_stats[k]) / 2 for k in forest_stats}
        
        logger.info(f"Epoch {epoch+1}, Train Loss: {train_stats['total_loss']:.4f}, "
                    f"Train cls_loss: {train_stats['total_cls_loss']:.4f}, "
                    f"Train reg_loss: {train_stats['total_reg_loss']:.4f}, "
                    f"Train acc_c: {train_stats['total_acc_c']:.4f}, "
                    f"Train acc_r: {train_stats['total_acc_r']:.4f}, "
                    f"Train miou: {train_stats['total_miou']:.4f}, "
                    f"Train rmse: {train_stats['total_rmse']:.4f}")
        
        wandb.log({"Train": train_stats, "epoch": epoch+1})

        forest_val_stats = validate(model, device, loss_fn, val_loader)
        city_val_stats = validate(model, device, loss_fn, target_val_loader, domain="City")
        
        val_stats = {k: (forest_val_stats[k] + city_val_stats[k]) / 2 for k in forest_val_stats}
        
        logger.info(f"Validation Loss: {val_stats['total_loss']:.4f}, "
                    f"Validation cls_loss: {val_stats['total_cls_loss']:.4f}, "
                    f"Validation reg_loss: {val_stats['total_reg_loss']:.4f}, "
                    f"Validation acc_c: {val_stats['total_acc_c']:.4f}, "
                    f"Validation acc_r: {val_stats['total_acc_r']:.4f}, "
                    f"Validation miou: {val_stats['total_miou']:.4f}, "
                    f"Validation rmse: {val_stats['total_rmse']:.4f}")
        
        wandb.log({"Validation": val_stats, "epoch": epoch+1})
        
        scheduler.step(val_stats['total_loss'])

        if val_stats['total_loss'] < best_val_loss:
            best_val_loss = val_stats['total_loss']
            torch.save(model.state_dict(), f"{checkpoint_path}/{name}_best.pth")
            logger.info(f"New best model saved at epoch {epoch+1}")

        if (epoch + 1) % 20 == 0:
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, f"{checkpoint_path}/{name}_checkpoint_{epoch+1}.pth")

        torch.save(model.state_dict(), f"{checkpoint_path}/{name}_last.pth")

    wandb.finish()
    
if __name__ =="__main__":
    main()