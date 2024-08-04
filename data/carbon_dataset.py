"""
Creates a Pytorch dataset to load the carbon dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# import cv2



class CarbonDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, #transform=None,
        cfg,
    ): 
        self.annotations = pd.read_csv(csv_file)
        self.cfg = cfg
        # self.img_dir = img_dir
        # self.label_dir = label_dir
        # self.transform = transform
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        out = dict()
        img_path = os.path.join(self.cfg.DATA_PATH, self.annotations.iloc[index, 0])
        img_SGRST_HIGH_path = os.path.join(self.cfg.DATA_PATH, self.annotations.iloc[index, 1])
        label_CRBN_QNTT_path = os.path.join(self.cfg.DATA_PATH, self.annotations.iloc[index, 2])
        label_tif_path = os.path.join(self.cfg.DATA_PATH, self.annotations.iloc[index, 3])


        i_i = Image.open(img_path)
        i_s = Image.open(img_SGRST_HIGH_path)       
        l_c = Image.open(label_CRBN_QNTT_path)
        l_t = Image.open(label_tif_path)
        
        img, label_cls, label_reg = self.normalize_concat(i_i, i_s, l_c, l_t)

        out["image"] = img
        out["label_cls"] = label_cls
        out["label_reg"] = label_reg       
        out["i_i_path"] = img_path
        out["i_s_path"] = img_SGRST_HIGH_path
        out["l_c_path"] = label_CRBN_QNTT_path
        out["l_t_path"] = label_tif_path


        return out

class CarbonDataset2(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, #transform=None,
        cfg,
        mode = "Valid"
        
    ): 
        self.annotations = pd.read_csv(csv_file, header=None, names=['Image_Path', 'SH_Path', 'Carbon_Path', 'GT_Path'])
        self.cfg = cfg
        self.mode = mode
        # self.img_dir = img_dir
        # self.label_dir = label_dir
        # self.transform = transform
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        out = dict()
        img_path = self.annotations.iloc[index, 0]
        img_SGRST_HIGH_path = self.annotations.iloc[index, 1]
        label_CRBN_QNTT_path = self.annotations.iloc[index, 2]
        label_tif_path = self.annotations.iloc[index, 3]

        if self.mode == "Valid":
            img_path = img_path.replace("training", "Validation")
            img_SGRST_HIGH_path = img_SGRST_HIGH_path.replace("training", "Validation")
            label_CRBN_QNTT_path = label_CRBN_QNTT_path.replace("training", "Validation")
            label_tif_path = label_tif_path.replace("training", "Validation")

        i_i = Image.open(img_path)
        i_s = Image.open(img_SGRST_HIGH_path)       
        l_c = Image.open(label_CRBN_QNTT_path).convert("L")
        l_t = Image.open(label_tif_path).convert("L")
        
        img, label_cls, label_reg = self.normalize_concat(i_i, i_s, l_c, l_t,self.cfg)

        out["image"] = img
        out["label_cls"] = label_cls
        out["label_reg"] = label_reg       
        out["i_i_path"] = img_path
        out["i_s_path"] = img_SGRST_HIGH_path
        out["l_c_path"] = label_CRBN_QNTT_path
        out["l_t_path"] = label_tif_path


        return out

    # def normalize_concat(self, i_i, i_s, l_c, l_t):
    @classmethod
    def normalize_concat(cls, i_i, i_s, l_c, l_t, cfg):
        
        i_i = np.array(i_i).astype(np.float32)/255.0
        ########if i_i == (512,512,4) in data50
        # i_i = i_i[...,0:3]
        #######################################
        
        i_s = np.array(i_s)
        i_s[np.isnan(i_s)] = 0
        # i_s = np.clip(np.array(i_s).astype(np.float32)/cfg.SGRST_CLIPPING, 0.0, 1.0) #임분고 영상 30이상은 클리핑
        i_s = np.clip(i_s.astype(np.float32)/cfg.SGRST_CLIPPING, 0.0, 1.0) #임분고 영상 30이상은 클리핑
        img = np.dstack((i_i,i_s))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # From HWC to CHW

        
        l_c = np.array(l_c)
        l_c[np.isnan(l_c)] = 0
        
        ####l_c 소수점버림##############################
        # l_c = np.trunc(l_c)
        
        
        # l_c = np.clip(np.array(l_c).astype(np.float32)/cfg.CARBON_CLIPPING, 0.0, 1.0)        
        # l_c = np.clip(l_c.astype(np.float32)/cfg.CARBON_CLIPPING, 0.0, 1.0)         
        l_c = np.clip((l_c.astype(np.float32)-cfg.CARBON_CLIPPING[0])/cfg.CARBON_CLIPPING[2], 0.0, 1.0).astype(np.float32)
        l_c = np.expand_dims(l_c, axis=-1)
        label_reg = torch.from_numpy(l_c)# From HWC to CHW

        l_t = np.array(l_t)
        
        ########if l_t == (512,512,2) in data50
        # l_t = l_t[...,0]
        # temp = np.full((512,512),255)
        # for k, v in cfg.label_mapping.items():
        #         temp[l_t == k] = v
        # l_t=temp                
        #######################################
        
        for k, v in cfg.label_mapping.items():
                l_t[l_t == k] = v
       
        l_t = np.expand_dims(l_t, axis=-1)
        label_cls = torch.from_numpy(l_t)# From HWC to CHW

        return img, label_reg, label_cls