from torch.utils import data
import os,  glob
from PIL import Image
import numpy as np
import torch
import logging
from data import augmentations

class NIADataset(data.Dataset):
    def __init__(self, root, image_size, split):

        self.root = root
        self.split = split
        self.files = {}
        self.image_size = image_size

        #self.images_base =os.path.join(self.root, self.split, "image", "*", "*", "*.png")
        self.images_base =os.path.join(self.root, self.split, "image", "*.png")
        self.mask_base = os.path.join(self.root, self.split, "label")

        self.files[self.split] = glob.glob(self.images_base)
        print(">>>>>>> Found %d %s images" % (len(self.files[split]), split))



        self.transform = None
        if split == "Training":
            self.transform = augmentations.Compose([augmentations.RandomVerticallyFlip(0.5),
                                                    augmentations.RandomHorizontallyFlip(0.5),
                                                    augmentations.RandomRotate(6),
                                                    augmentations.AdjustBrightness(0.2)])

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        out = dict()

        logging.getLogger('PIL').setLevel(logging.WARNING)

        img_path = self.files[self.split][index].rstrip()
        dir = os.path.dirname(img_path).split("/")        
        name = os.path.basename(img_path).split(".")[0].split("_")
        mask_name = name[0]+"_"+name[1]+"_"+name[2]+"_gt_"+name[3]+".png"
        #mask_path = os.path.join(self.mask_base, dir[-2], dir[-1], mask_name)
        mask_path = os.path.join(self.mask_base, mask_name)


        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        #image crop
        #org_img_size = list(image.size)
        #image = image.crop((170, 0, int(org_img_size[0]), int(org_img_size[1])))
        #mask = mask.crop((170, 0, int(org_img_size[0]), int(org_img_size[1])))

        image, mask = self.resize(image, mask)

        #if self.transform is not None:
        #    image, mask = self.transform(image, mask)

        image, mask = self.normalize(image, mask)

        out["image"] = image
        out["mask"] = mask
        out["img_path"] = img_path
        out["mask_path"] = mask_path

        return out

    def resize(self, image, mask):
        image = image.resize((self.image_size, self.image_size), resample=Image.LANCZOS)  # uint8 with RGB mode
        mask = mask.resize((self.image_size, self.image_size))
        return image, mask

    def normalize(self, image, mask):
        #image = np.array(image).astype(np.float64)
        #image = np.clip(image - np.median(image) + 127, 0, 255)/255.0
        image = np.array(image).astype(np.float64)/255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # From HWC to CHW

        mask = np.array(mask).astype(np.float64)/255.0
        mask = np.expand_dims(mask, axis=-1)
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()# From HWC to CHW

        return image, mask


