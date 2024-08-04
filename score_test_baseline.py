import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.unet import UNet_carbon
from dataset_segwithcarbon import CarbonDataset, CarbonDataset_csv
from model.util import select_device, mix_patch
from tqdm import tqdm
# from model.metrics import CarbonLoss , CarbonLossWithRMSE
import os
import logging
from cmath import nan
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config.param_parser import InferenceParser
from evaluate import corr_wCla, r_square_wCla
from utils.utils import make_directory
from model.build import build_model
from data.carbon_dataset import CarbonDataset2
from data.build import build_data_loader
from model.metrics import CarbonLossWithRMSE, CarbonLoss
from config_mf import CONFIGURE
from evaluate import corr, r_square, corr_wZero, r_square_wZero
def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask] +
        label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist
def compute_miou(hist):
    with torch.no_grad():
        iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))
        miou = torch.nanmean(iou)
    return miou

def batch_miou(label_preds, label_trues, num_class, device):
    label_trues = label_trues.long()  # 정수 타입으로 변환
    label_preds = label_preds.long()  # 정수 타입으로 변환
    
    """Calculate mIoU for a batch of predictions and labels"""
    hist = torch.zeros((num_class, num_class), device=device)  # 디바이스 지정
    for lt, lp in zip(label_trues, label_preds):
        lt = lt.to(device)  # 디바이스 이동
        lp = lp.to(device)  # 디바이스 이동
        hist += fast_hist(lt.flatten(), lp.flatten(), num_class)
    return compute_miou(hist)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class CarbonLoss(nn.Module):
    def __init__(self,num_classes=4, weight=None, size_average=True, cls_lambda=50, reg_lambda=0.005, ignore_label=255):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.cls_lambda = cls_lambda
        self.reg_lambda = reg_lambda
        self.num_classes = num_classes

    def forward(self, input_cls,input_reg, target_cls, target_reg):
        # pred = input.view(-1)
        # truth = target.view(-1)
        
        cls_loss = self.ce(input_cls, target_cls.type(torch.long).squeeze())
        _, input_cls_pred = torch.max(input_cls, dim=1)
        miou = batch_miou(input_cls_pred, target_cls, self.num_classes, torch.device("cpu"))
        
        reg_loss = self.mse(
            torch.flatten(input_reg, end_dim=-2),
            torch.flatten(target_reg, end_dim=-2)
        )
        # Image.fromarray( (np.clip((input_reg[3,...].squeeze().detach().cpu().numpy()*255), 0,255)).astype(np.uint8)).save("/root/work/src/carbon/debug3.png")  #for debug
        input_reg = input_reg.squeeze()
        target_reg = target_reg.squeeze()

        corr_sum=0
        r_sum =0
        rmse_sum = 0
        idx = 0
        for i in range(len(input_reg)):
            ir = input_reg[i,...].cpu().detach().numpy()
            tr = target_reg[i,...].cpu().detach().numpy()
            if np.count_nonzero(tr) == 0:
                continue          
            corr_res = corr_wZero(ir,tr)
            r_res = r_square_wZero(ir,tr)
            rmse_res = np.sqrt(np.mean((ir - tr)**2))
            if np.isnan(corr_res) or np.isnan(r_res) or np.isnan(rmse_res):           
                continue

            corr_sum = corr_sum + corr_res
            r_sum = r_sum + r_res
            rmse_sum += rmse_res
            idx = idx +1
            
        acc_c = corr_avg = corr_sum/idx
        acc_r = r_avg = r_sum/idx
        avg_rmse = rmse_sum / max(idx, 1)
        # acc_c = corr(torch.flatten(input_reg, end_dim=-2).cpu().detach().numpy(), torch.flatten(target_reg, end_dim=-2).cpu().detach().numpy())
        # acc_r = r_square(torch.flatten(input_reg, end_dim=-2).cpu().detach().numpy(), torch.flatten(target_reg, end_dim=-2).cpu().detach().numpy())

        total_loss = self.cls_lambda * cls_loss + self.reg_lambda * reg_loss

        return total_loss, cls_loss, reg_loss, acc_c, acc_r , miou, avg_rmse


def validate(model, device, loss_fn, loader, domain="Forest"):
    model.eval()
    val_stats = {
        'total_loss': AverageMeter(), 'cls_loss': AverageMeter(), 'reg_loss': AverageMeter(),
        'acc_c': AverageMeter(), 'acc_r': AverageMeter(), 'miou': AverageMeter(), 'rmse': AverageMeter(),
        'corr': AverageMeter(), 'r_square': AverageMeter()
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating {domain}"):
            x = batch["image"].to(device)
            carbon = batch["label_reg"].to(device).squeeze(3)  # [8, 512, 512]
            gt = batch["label_cls"].to(device).squeeze(3).long()  # [8, 512, 512]
            
            # print("Input shape:", x.shape)
            # print("Carbon label shape:", carbon.shape)
            # print("GT label shape:", gt.shape)
            
            gt_pred, carbon_pred = model(x)
            gt_pred.to('cpu')
            carbon_pred.to('cpu')
            # print("GT prediction shape:", gt_pred.shape)
            # print("Carbon prediction shape:", carbon_pred.shape)

            total_loss, cls_loss, reg_loss, acc_c, acc_r, miou, rmse = loss_fn(gt_pred, carbon_pred,gt.squeeze(1) , carbon)
            
            # Calculate correlation and R-squared
            corr_val = corr_wCla(carbon_pred.cpu().numpy(), carbon.cpu().numpy(), gt.cpu().numpy())
            r_square_val = r_square_wCla(carbon_pred.cpu().numpy(), carbon.cpu().numpy(), gt.cpu().numpy())
            
            val_stats['total_loss'].update(total_loss.item(), x.size(0))
            val_stats['cls_loss'].update(cls_loss.item(), x.size(0))
            val_stats['reg_loss'].update(reg_loss.item(), x.size(0))
            val_stats['acc_c'].update(acc_c, x.size(0))
            val_stats['acc_r'].update(acc_r, x.size(0))
            val_stats['miou'].update(miou, x.size(0))
            val_stats['rmse'].update(rmse, x.size(0))
            val_stats['corr'].update(corr_val, x.size(0))
            val_stats['r_square'].update(r_square_val, x.size(0))

    return val_stats

def validate_and_visualize(args):
    cfg = CONFIGURE(args.image_type)
    make_directory(cfg.RESULT_OUT_DIR)
    model = build_model(args.net, channel=cfg.NUM_CLASSES) # class_num = channel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('On which device we are on : {}'.format(device))
    model.to(device)    # move the model to GPU
    
    ckpt_file_path = cfg.MODEL_PATH
    print("loading model...")
    state_dict = torch.load(ckpt_file_path, map_location="cpu")
    new_state_dict = {}
    for key in state_dict['model']:
        new_key = key.replace('module.','')
        new_state_dict[new_key] = state_dict['model'][key]
    model.load_state_dict(new_state_dict)
    print(f" => loaded checkpoint {ckpt_file_path}")

    del new_state_dict
    torch.cuda.empty_cache()
    model.eval()

    # val_loader = build_data_loader("carbon", base_dir=cfg.DATA_PATH,split=cfg.VAL_CSV,image_size=cfg.IMAGE_SIZE,
    #                                batch_size= 8, num_workers=args.num_workers, local_rank=args.local_rank, cfg=cfg, shuffle=False)
    val_dataset = CarbonDataset2("val_AP25_Forest_IMAGE.csv", cfg)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=8,
    )
    criterion = CarbonLoss(num_classes=cfg.NUM_CLASSES, cls_lambda=1, reg_lambda=0.0005)

    val_stats = validate(model, device, criterion, val_loader, domain=args.image_type)

    print(f"Validation Loss: {val_stats['total_loss'].avg:.4f}, "
          f"Validation Acc (C): {val_stats['acc_c'].avg:.4f}, "
          f"Validation Acc (R): {val_stats['acc_r'].avg:.4f}, "
          f"Validation mIoU: {val_stats['miou'].avg:.4f}, "
          f"Validation RMSE: {val_stats['rmse'].avg:.4f}, "
          f"Validation Correlation: {val_stats['corr'].avg:.4f}, "
          f"Validation R-squared: {val_stats['r_square'].avg:.4f}")
device = select_device()
if __name__ == '__main__':
    parser = InferenceParser()
    args = parser.parse_args()
    validate_and_visualize(args)

