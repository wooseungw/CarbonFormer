import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_image_paths(path):
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.tif') or filename.endswith('.png'):
            image_paths.append(os.path.join(path, filename).replace('\\', '/'))
    if len(image_paths) != 0:
        print(path, "Done.",end='/')
    print()
    return image_paths

class Mapping():
    def __init__(self, folder_path):
        if "AP10_City" in folder_path:
            self.label_name = "AP10_City"
            # self.label_mapping={0:   0,                      
            #                     110: 1,
            #                     120: 2,
            #                     130: 3,
            #                     140: 4,
            #                     210: 5,
            #                     220: 6,
            #                     230: 7, 
            #                     190: 8,
            #                     255: 0}
        if "AP25_City" in folder_path:
            self.label_name = "AP25_City"
            # self.label_mapping={0:   0,                      
            #                     110: 1,
            #                     120: 2,
            #                     130: 3,
            #                     140: 4,
            #                     210: 5,
            #                     220: 6,
            #                     230: 7, 
            #                     190: 8,
            #                     255: 0}
        if "AP10_Forest" in folder_path:
            self.label_name = "AP10_Forest"
            # self.label_mapping={0:  0,                      
            #                     110: 1,
            #                     120: 2,
            #                     130: 3,
            #                     140: 4,
            #                     150: 5,
            #                     190: 6, 
            #                     255: 0}
        if "AP25_Forest" in folder_path:
            self.label_name = "AP25_Forest"
            # self.label_mapping={0: 0,                      
            #                     110:1,
            #                     120:2,
            #                     130:3,
            #                     140: 4,
            #                     150: 5,
            #                     190: 6, 
            #                     255: 0}
        if "SN10_Forest" in folder_path:
            self.label_name = "SN10_Forest"
            # self.label_mapping={0:  0,                      
            #                     140: 1,
            #                     150: 2,
            #                     190: 3, 
            #                     255: 0}
        self.label_mapping={0:   0,                      
                                110: 1,
                                120: 1,
                                130: 1,
                                140: 2,
                                150: 1,
                                210: 2,
                                220: 3,
                                230: 3, 
                                190: 3,
                                255: 0}
    def __call__(self, img):
        return self.gt_mapping(img)

    def gt_mapping(self, img):
        # PIL 이미지를 NumPy 배열로 변환
        image_np = np.array(img)
        
        # 출력 배열 초기화 (입력 이미지와 동일한 크기)
        mapped_image_np = np.zeros_like(image_np)
        
        # 라벨 매핑 적용
        for original_label, mapped_label in self.label_mapping.items():
            mapped_image_np[image_np == original_label] = mapped_label
        
        # 매핑된 NumPy 배열을 다시 PIL 이미지로 변환 (필요한 경우)
        mapped_image = Image.fromarray(mapped_image_np)
        
        return mapped_image
    
class CarbonDataset(Dataset):
    def __init__(self, folder_path, image_transform=None,sh_transform=None,label_transform=None, mode = "Train"):
        if mode == "Valid":
            folder_path = folder_path.replace("Training",'Validation')
            print("Validation Set")
        else:
            print("Training Set")
        #이미지와 임분고 리스트 반환
        self.image_paths = get_image_paths(folder_path)
        self.sh_paths = get_image_paths(folder_path.replace("IMAGE","SH"))
        #탄소량과 Gt리스트 반환
        folder_path = folder_path.replace("image",'label')
        self.carbon_paths = get_image_paths(folder_path.replace("IMAGE","Carbon"))
        self.gt_paths = get_image_paths(folder_path.replace("IMAGE","GT"))
        self.image_transform = image_transform
        self.sh_transform = sh_transform
        self.label_transform = label_transform
        self.Mapping = Mapping(folder_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        sh_path = self.sh_paths[idx]
        sh = Image.open(sh_path).convert('L')
        
        carbon_path = self.carbon_paths[idx]
        carbon = Image.open(carbon_path)
        
        gt_paths = self.gt_paths[idx]
        gt = Image.open(gt_paths).convert('L')
        gt = self.Mapping(gt)
        #print(gt)
        
        if self.image_transform:
            image = self.image_transform(image)
            
        if self.sh_transform:
            sh = self.sh_transform(sh)
            
        if self.label_transform:
            carbon = self.label_transform(carbon)
            gt = self.label_transform(gt)
            
        gt = torch.tensor(np.array(gt), dtype=torch.float32).unsqueeze(0)
        carbon = torch.tensor(np.array(carbon), dtype=torch.float32).unsqueeze(0)
        
        return image, sh , carbon , gt

class CarbonDataset_csv(Dataset):
    def __init__(self, csv_file, image_transform=None, sh_transform=None, label_transform=None, mode="Train"):
        self.mode = mode
        self.data = pd.read_csv(csv_file, header=None, names=['Image_Path', 'SH_Path', 'Carbon_Path', 'GT_Path'])        

        
        self.image_transform = image_transform
        self.sh_transform = sh_transform
        self.label_transform = label_transform
        self.Mapping = Mapping(self.data.iloc[0, 0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        sh_path = self.data.iloc[idx, 1]
        carbon_path = self.data.iloc[idx, 2]
        gt_path = self.data.iloc[idx, 3]
        if self.mode == "Valid":
            img_path = img_path.replace("Training", "Validation")
            sh_path = sh_path.replace("Training", "Validation")
            carbon_path = carbon_path.replace("Training", "Validation")
            gt_path = gt_path.replace("Training", "Validation")
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        sh = Image.open(sh_path).convert('L')  # SH 이미지를 그레이스케일로 변환
        carbon = Image.open(carbon_path).convert('L')
        gt = Image.open(gt_path).convert('L')

        # GT 이미지에 매핑 적용
        gt = self.Mapping(gt)
        
        # 변환 적용
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.sh_transform:
            sh = self.sh_transform(sh)
        
        if self.label_transform:
            carbon = self.label_transform(carbon)
            gt = self.label_transform(gt)
        
        # 텐서로 변환
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(np.array(image).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
        if not isinstance(sh, torch.Tensor):
            sh = torch.tensor(np.array(sh), dtype=torch.float32).unsqueeze(0) / 255.0
        if not isinstance(carbon, torch.Tensor):
            carbon = torch.tensor(np.array(carbon), dtype=torch.float32).unsqueeze(0) / 255.0
        if not isinstance(gt, torch.Tensor):
            gt = torch.tensor(np.array(gt), dtype=torch.float32).unsqueeze(0)
        
        
        return image, sh, carbon, gt
  
# 시각화 코드 예시
def imshow(tensor, title=None):
    image = tensor.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

if __name__ == "__main__":
    # Set the folder path for the dataset
    folder_paths = ['dataset/training/image/AP25_City_IMAGE','dataset/training/image/AP25_Forest_IMAGE']
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    transform_label = transforms.Compose([transforms.Resize((512, 512))])
    # Create an instance of the CustomImageDataset class
    dataset = CarbonDataset_csv("train_AP25_Forest_IMAGE.csv", transform, transform, transform_label, mode="Train")  
    print(len(dataset))
  
  
    # Create a data loader for the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    sample_index = 0
    # Iterate over the dataset and print the images and labels
    for image, sh, carbon, gt in dataloader:
        # image
        print("Image:", image.shape, image.type())
        print(image.min(), image.max())
        print(image)
        plt.subplot(2, 2, 1)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.title("1. Image")
        plt.xticks([])
        plt.yticks([])
        
        #sh
        print("2. SH:", sh.shape, sh.type())
        print(sh.min(), sh.max())
        print(sh)
        plt.subplot(2, 2, 2)
        plt.imshow(sh.squeeze())
        plt.title("2. SH")
        plt.xticks([])
        plt.yticks([])
        
        # carbon
        print("3. Carbon:", carbon.shape, carbon.type())
        print(carbon.min(), carbon.max())
        print(carbon)
        plt.subplot(2, 2, 3)
        plt.imshow(carbon.squeeze())
        plt.title("3. Carbon")
        plt.xticks([])
        plt.yticks([])
        
        # gt
        print("4. GT:", gt.shape, gt.type())
        print(gt.min(), gt.max())
        print(carbon.shape)
        print(gt)
        plt.subplot(2, 2, 4)
        plt.imshow(gt.squeeze())
        plt.title("4. GT")
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        plt.show()

        # print(image.shape, image.type())

        # print(carbon.shape, carbon.type())
        # print(gt.shape, gt.type())
        


