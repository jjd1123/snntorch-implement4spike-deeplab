from torchvision.datasets import VOCSegmentation
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
import torch
import numpy as np
import torchvision

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
def voc_rand_crop(feature,label,height,width):
    rect = torchvision.transforms.RandomCrop.get_params(feature,(height,width))
    feature = torchvision.transforms.functional.crop(feature,*rect)
    label = torchvision.transforms.functional.crop(label,*rect)
    return feature,label

def voc_colormap2label():
    colormaplabel = torch.zeros(256 ** 3, dtype=torch.long)
    colormaplabel[:] =255
    for i, colormap in enumerate(VOC_COLORMAP):
        idx = (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        colormaplabel[idx] = i
    return colormaplabel

def voc_label_indices(label, colormaplabel):
    label = label
    idxs = [(label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]]
    return colormaplabel[idxs]

class vocDataset(VOCSegmentation):
    
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"
    def __init__(self, root: str, year: str = "2012", image_set: str = "train", download: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.colormap2label = voc_colormap2label()
        self.train = image_set
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target = np.array(target,dtype=np.int32)
        return img, voc_label_indices(target, self.colormap2label)