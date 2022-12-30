from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset
import os
class augvocDataset(Dataset):
    def __init__(self,img_dir, im_set = "train", transform=None, target_transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        target_root = os.path.join(img_dir,"aug_cls")
        img_root = os.path.join(img_dir,"aug_img")
        if im_set == "train":
            txt_path = os.path.join(img_dir,"train.txt")
            with open(os.path.join(txt_path), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
            self.images = [os.path.join(img_root, x.split(' ')[0] ) for x in file_names]
            self.targets = [os.path.join(target_root, x.split(' ')[1] ) for x in file_names]
        else:
            txt_path = os.path.join(img_dir,"val.txt")  
            with open(os.path.join(txt_path), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
            self.images = [os.path.join(img_root, x + ".jpg") for x in file_names]
            self.targets = [os.path.join(target_root, x + ".png") for x in file_names]  
            
            assert len(self.images) == len(self.targets)
                
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index]).convert("P")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = np.array(target)
        return img,target
    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    from torchvision import transforms
    batch_size = 16
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
    train_img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
    tar_transform = transforms.Compose([transforms.Resize([64,64])])
    test_tar_transform = transforms.Compose([transforms.Resize([64,64])])
    dataset = augvocDataset("/mnt/xlancefs/home/zsx66/dataset/snn/data",transform=train_img_transform,target_transform=tar_transform)
    test = augvocDataset("/mnt/xlancefs/home/zsx66/dataset/snn/data","val",transform=train_img_transform,target_transform=tar_transform)
    for i,p in iter(dataset):
        print(i.shape,p.shape)
        break
    for i,p in iter(test):
        print(i.shape,p.shape)
        break