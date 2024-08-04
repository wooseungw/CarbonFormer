import os
import random

import numpy as np
import cv2
from glob import glob
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse


#python data/proc_argumentation_dataset.py --split Test


def read_image(imagefile, grayscale=False):
    if grayscale == True:
        image = cv2.imread(imagefile)
        #image = np.expand_dims(image, -1)
    else:
        image = cv2.imread(imagefile)
    return image


def save_image(image, mask, path, binary=True):
    image = np.array(image)
    if binary == True:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path[0], image)
    cv2.imwrite(path[1], mask)

def concat_images(images, rows, cols):
    _, h, w, _ = images.shape
    images = images.reshape((rows, cols, h, w, 3))
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape((rows * h, cols * w, 3))
    return images

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

def subtract(image):
    image = image / 255
    return image

def resize(image, size):
    size = check_size(size)
    image = cv2.resize(image, size)
    return image

def center_crop(image, mask, crop_size, size):
    h, w, _ = image.shape
    crop_size = check_size(crop_size)
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def random_crop(image, mask, crop_size, size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def horizontal_flip(image, mask, size):
    image = image[:, ::-1, :]
    mask = mask[:, ::-1, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def vertical_flip(image, mask, size):
    image = image[::-1, :, :]
    mask = mask[::-1, :, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def scale_augmentation(image, mask, scale_range, crop_size, size):
    scale_size = np.random.randint(*scale_range)
    image = cv2.resize(image, (scale_size, scale_size))
    mask = cv2.resize(mask, (scale_size, scale_size))
    image, mask = random_crop(image, mask, crop_size, size)
    return image, mask

def random_rotation(image, mask, size, angle_range=(0, 90)):
    h1, w1, _ = image.shape
    h2, w2, _ = mask.shape

    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h1, w1))

    mask = rotate(mask, angle)
    mask = resize(mask, (h2, w2))

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def cutout(image_origin, mask_origin, mask_size, size, mask_value='mean'):
    image = np.copy(image_origin)
    mask = np.copy(mask_origin)

    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    mask[top:bottom, left:right, :].fill(0)

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def brightness_augment(img, mask, size, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)

    image = resize(rgb, size)
    mask = resize(mask, size)

    return image, mask

def rgb_to_grayscale(img, mask, size):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = [img, img, img]
    img = np.transpose(img, (1, 2, 0))

    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask

def blur_averaging(img, mask, filter_size, size):
    img = cv2.blur(img,(filter_size,filter_size))
    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask

def blur_bilateral(img, mask, filter_size1, filter_size2, filter_size3, size):
    img = cv2.bilateralFilter(img, filter_size1, filter_size2, filter_size3)
    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask

def blur_gaussian(img, mask, filter_size, size):
    img = cv2.GaussianBlur(img, (filter_size,filter_size), 0)
    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask

def lens_distortion(img, mask, exp, scale):
    rows, cols = img.shape[:2]
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)

    mapx = 2 * mapx / (cols - 1) - 1
    mapy = 2 * mapy / (rows - 1) - 1

    r, theta = cv2.cartToPolar(mapx, mapy)
    r[r < scale] = r[r < scale] ** exp

    mapx, mapy = cv2.polarToCart(r, theta)


    mapx = ((mapx + 1) * cols - 1) / 2
    mapy = ((mapy + 1) * rows - 1) / 2

    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    mask = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)

    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask


def create_dir(name):
    try:
        os.mkdir(name)
    except:
        pass


def create_path(path, split, save_base_path):
    path1 = path.split(split + "/")[1].split("/")
    new_path = save_base_path
    for i in range(1):
        #print("new_path = ", new_path)
        #print("path1 = ", path1)
        new_path = os.path.join(new_path, path1[i])
        if not os.path.exists(new_path):
            os.mkdir(os.path.join(new_path))
    return new_path, path1[-1]

parser = argparse.ArgumentParser(description='Process data argumentation.....')
parser.add_argument('--split', type=str,   default='Training', help="dataset name")

if __name__ == '__main__':
    #size = (256, 256)
    #crop_size = (300, 300)
    args = parser.parse_args()
    split = args.split

    path = "/workspace/data/"
    dataset_name = "revised_NIA"
    full_path = os.path.join(path, dataset_name)

    new_path = path
    dataset_name = "NIA_arg"
    new_full_path = os.path.join(new_path, dataset_name)
    #create_dir(new_full_path)

    save_base_path = os.path.join(new_full_path, split)
    #valid_path = os.path.join(new_full_path, "Validation")
    #test_path = os.path.join(new_full_path, "Test")

    print("save_base_path = ", save_base_path)
    if not os.path.exists(new_full_path):
        os.mkdir(new_full_path)
    if not os.path.exists(save_base_path):
        os.mkdir(save_base_path)
    if not os.path.exists(os.path.join(save_base_path, "image")):
        os.mkdir(os.path.join(save_base_path, "image"))
    if not os.path.exists(os.path.join(save_base_path, "label")):
        os.mkdir(os.path.join(save_base_path, "label"))


    files = {}
    files[split] = glob(os.path.join(full_path, split, "image/", "*.png"))
    mask_base = os.path.join(full_path, split, "label")
    
    print(files)

    crop_size = (300, 300)
    size = (256, 256)

    for idx, p in tqdm(enumerate(files[split]), total=len(files[split])):

        img_path = files[split][idx].rstrip()
        dir = os.path.dirname(img_path).split("/")
        mask_path = os.path.join(mask_base, os.path.basename(img_path).replace(".png", "_gt.png"))
        
        if os.path.exists(img_path) and os.path.exists(img_path):
            image = read_image(img_path)
            mask = read_image(mask_path, grayscale=True)
            #print("img_path = ", img_path)

            img_height = image.shape[0]
            img_width = image.shape[1]

            # crop
            image = image[35:img_height-35, 170:img_width]
            mask = mask[35:img_height - 35, 170:img_width]
            #
            #cv2.imwrite("image.jpg", image)
            #cv2.imwrite("mask.jpg", mask)
            if split == "Validation":
                image1, mask1 = horizontal_flip(image, mask, size)
                image2, mask2 = vertical_flip(image, mask, size)
                                
                image = resize(image, size)
                mask = resize(mask, size)

                all_images = [image, image1, image2]
                all_masks = [mask, mask1, mask2]     

            elif split== "Training":
                # center_crop
                image1, mask1 = center_crop(image, mask, crop_size, size)
                image2, mask2 = random_crop(image, mask, crop_size, size)
                image3, mask3 = horizontal_flip(image, mask, size)
                image4, mask4 = vertical_flip(image, mask, size)
                image5, mask5 = scale_augmentation(image, mask, (512, 768), crop_size, size)
                image6, mask6 = random_rotation(image, mask, size)
                image7, mask7 = cutout(image, mask, 256, size)
                ## Extra Cropping
                image8, mask8 = random_crop(image, mask, crop_size, size)
                image9, mask9 = random_crop(image, mask, crop_size, size)
                ## Extra Scale Augmentation
                image10, mask10 = scale_augmentation(image, mask, (540, 820), crop_size, size)
                image11, mask11 = scale_augmentation(image, mask, (720, 1024), crop_size, size)
                ## Extra Rotation
                image12, mask12 = random_rotation(image, mask, size)
                image13, mask13 = random_rotation(image, mask, size)
                ## Brightness
                image14, mask14 = brightness_augment(image, mask, size, factor=0.3)
                image15, mask15 = brightness_augment(image, mask, size, factor=0.6)
                image16, mask16 = brightness_augment(image, mask, size, factor=0.9)
                ## More Rotation
                image17, mask17 = random_rotation(image, mask, size)
                image18, mask18 = random_rotation(image, mask, size)
                ## More Random Crop
                image19, mask19 = random_crop(image, mask, crop_size, size)
                image20, mask20 = random_crop(image, mask, crop_size, size)
                ## More Cutout
                image21, mask21 = cutout(image, mask, 256, size)
                image22, mask22 = cutout(image, mask, 256, size)
                ## Grayscale
                image23, mask23 = rgb_to_grayscale(image, mask, size)
                image24, mask24 = rgb_to_grayscale(image1, mask1, size)
                image25, mask25 = rgb_to_grayscale(image2, mask2, size)
                image26, mask26 = rgb_to_grayscale(image3, mask3, size)
                image27, mask27 = rgb_to_grayscale(image4, mask4, size)
                image28, mask28 = rgb_to_grayscale(image5, mask5, size)
                image29, mask29 = rgb_to_grayscale(image15, mask15, size)
                image30, mask30 = rgb_to_grayscale(image16, mask16, size)
                # blur
                image31, mask31 = blur_averaging(image, mask, filter_size=10, size=size)
                image32, mask32 = blur_averaging(image, mask, filter_size=5, size=size)
                image33, mask33 = blur_averaging(image, mask, filter_size=20, size=size)
                image34, mask34 = blur_averaging(image, mask, filter_size=15, size=size)

                image35, mask35 = blur_bilateral(image, mask, filter_size1=7, filter_size2=75, filter_size3=75,
                                                 size=size)
                image36, mask36 = blur_bilateral(image, mask, filter_size1=11, filter_size2=10, filter_size3=10,
                                                 size=size)
                image37, mask37 = blur_bilateral(image, mask, filter_size1=3, filter_size2=100, filter_size3=100,
                                                 size=size)
                image38, mask38 = blur_bilateral(image, mask, filter_size1=21, filter_size2=30, filter_size3=30,
                                                 size=size)

                image39, mask39 = blur_gaussian(image, mask, filter_size=5, size=size)
                image40, mask40 = blur_gaussian(image, mask, filter_size=3, size=size)
                image41, mask41 = blur_gaussian(image, mask, filter_size=7, size=size)
                image42, mask42 = blur_gaussian(image, mask, filter_size=11, size=size)

                # lens distortion
                image43, mask43 = lens_distortion(image, mask, 1.5, 1)
                image44, mask44 = lens_distortion(image, mask, 0.5, 1)
                image45, mask45 = lens_distortion(image, mask, 1.2, 1)
                image46, mask46 = lens_distortion(image, mask, 0.8, 1)

                ## Original image and mask
                image = resize(image, size)
                mask = resize(mask, size)

                ## All images and masks
                all_images = [image, image1, image2, image3, image4, image5, image6, image7,
                              image8, image9, image10, image11, image12, image13, image14, image15, image16,
                              image17, image18, image19, image20, image21, image22,
                              image23, image24, image25, image26, image27, image28, image29, image30,
                              image31, image32, image33, image34, image35, image36, image37, image38, image39,
                              image40, image41, image42, image43, image44, image45, image46
                              ]
                all_masks = [mask, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8,
                             mask9, mask10, mask11, mask12, mask13, mask14, mask15, mask16,
                             mask17, mask18, mask19, mask20, mask21, mask22,
                             mask23, mask24, mask25, mask26, mask27, mask28, mask29, mask30,
                             mask31, mask32, mask33, mask34, mask35, mask36, mask37, mask38, mask39, mask40,
                             mask41, mask42, mask43, mask44, mask45, mask46
                             ]
              
            new_image_path, imgname = create_path(img_path, split, save_base_path)
            
            #print("new_image_path= ", new_image_path)
            
            new_mask_path, maskname = create_path(mask_path, split, save_base_path)
            
            

            for j in range(len(all_images)):
                ver_name = "_%06d" % j + ".png"
                save_img_path = os.path.join(new_image_path, imgname.split(".png")[0] + ver_name)
                save_mask_path = os.path.join(new_mask_path, maskname.split(".png")[0] + ver_name)
                print(">> save_img_path = ", save_img_path)
                print("save_mask_path = ", save_mask_path)

                img = all_images[j]
                msk = all_masks[j]
                path = [save_img_path, save_mask_path]
                save_image(img, msk, path)
        
       
