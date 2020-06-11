import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import platform

class MY_data(data.Dataset):
    def __init__(self, root,image_size=512,mode='train',augmentation_prob=0.6):
        """Initializes image paths and preprocessing module."""
        if platform.system() == "Windows":
            self.root = root
            # GT : Ground Truth
            self.GT_paths = root+'_GT\\'
        else:
            self.root = root+'/'
            # GT : Ground Truth
            self.GT_paths = root+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_GT_paths = list(map(lambda x: os.path.join(self.GT_paths, x), os.listdir(self.GT_paths)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        #print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        GT_path = self.image_GT_paths[index]
        image = Image.open(image_path)
        GT = Image.open(GT_path)

        # Transform = T.Compose([
        #     T.Resize(self.image_size),
        #     T.CenterCrop(self.image_size),
        #     T.ToTensor(),
        #     T.Normalize([0.5],[0.5])
        # ])
        Transform = []
        p_transform = random.random()
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]                          
            
            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))

            Transform = T.Compose(Transform)

            image = Transform(image)
            GT = Transform(GT)

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            # NO working RGB->HSV
            # Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
            # image = Transform(image)

        Transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize([0.5],[0.5])
        ])
		
        image = Transform(image)
        GT = Transform(GT)

        return image , GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    datasets = MY_data(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset    = datasets,
    							  batch_size = batch_size,
    							  shuffle    = True,
    							  num_workers= num_workers)
    return data_loader