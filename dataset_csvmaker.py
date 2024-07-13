import csv
import os
from torch.utils.data import Dataset, DataLoader

def get_image_paths(path):
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.tif') or filename.endswith('.png'):
            image_paths.append(os.path.join(path, filename).replace('\\', '/'))
    if len(image_paths) != 0:
        print(path, "Done.")
    return image_paths

class MakeCSV():
    def __init__(self, folder_path) -> None:
        self.image_paths = []
        self.sh_paths = []
        self.carbon_paths = []
        self.gt_paths = []
        
        self.csv_file_path = f"Dataset/{folder_path.split("/")[-1]}.csv" if os.path.exists("Dataset") else "dataset.csv"

        self.image_paths = get_image_paths(folder_path)
        self.image_paths.sort()
        self.sh_paths = get_image_paths(folder_path.replace("IMAGE", "SH"))
        self.sh_paths.sort()
        folder_path = folder_path.replace("image", "label")
        self.carbon_paths = get_image_paths(folder_path.replace("IMAGE", "Carbon"))
        self.carbon_paths.sort()
        self.gt_paths = get_image_paths(folder_path.replace("IMAGE", "GT"))
        self.gt_paths.sort()
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image_Path", "SH_Path", "Carbon_Path", "GT_Path"])
            for i in range(len(self.image_paths)):
                writer.writerow([self.image_paths[i], self.sh_paths[i], self.carbon_paths[i], self.gt_paths[i]])
        
        print("CSV file created successfully!")

if __name__ == "__main__":
    folder_paths = ['dataset/training/image/AP10_City_IMAGE',
                    'dataset/training/image/AP10_Forest_IMAGE',
                    'dataset/training/image/AP25_City_IMAGE',
                    'dataset/training/image/AP25_Forest_IMAGE',
                    'dataset/training/image/SN10_Forest_IMAGE',
                    ]
    for path in folder_paths:
        MakeCSV(path)