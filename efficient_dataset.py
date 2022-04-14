import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from skimage.transform import rotate, rescale, resize

# from retina_transform import foveat_img
import sys
sys.path.append('./transformations')
from transformations import SalienceSamplingOld as SalienceSampling, LogPolar, NRandomCrop, Compose, Foveate

from torch.profiler import profile, record_function, ProfilerActivity

class TransformedData(datasets.ImageFolder):
    def __init__(self, data_path, salience_path, crop_size, max_rotation, lp, lp_out_shape, augmentation, points, inversion, count = 1):
        super().__init__(data_path)

        self.count = count

        self.augmentation = augmentation
        self.points = points
        self.crop_size = crop_size
        self.salience_path = salience_path

        trans = []

        self.tensorize = transforms.ToTensor()
        self.normalize = transforms.Normalize( mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225) )

        if augmentation == 'salience':
            self.transform = SalienceSampling(self.points, self.crop_size, self.salience_path)
        elif augmentation == 'random': 
            self.transform = NRandomCrop(self.points, crop_size)
        else:
            self.transform = transforms.Resize(crop_size)
        trans.append(self.transform)

        if inversion:
            self.rotate = transforms.RandomRotation((180,180))
        else:
            self.rotate = transforms.RandomRotation(max_rotation)
        trans.append(self.rotate)

#        self.fov = Foveate(self.crop_size)
#        trans.append(self.fov)

        if lp:
            self.lp = LogPolar(output_shape = lp_out_shape)
        trans.append(self.lp)

        self.augment = Compose(trans)


    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)

        image = self.tensorize(image) # tensorize
        image = self.normalize(image) # normalize

        if not self.transform:
            return path, image, SalienceSampling.getSalienceMap(self.salience_path, path), target

        image = torch.stack([self.augment(image, path) for _ in range(self.count)])

        return image, target

