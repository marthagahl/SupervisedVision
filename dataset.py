import os 
from collections import namedtuple
from functools import partial
import csv
import torch.nn as nn
import numpy as np
import torch
import PIL
import torch.utils.data
from torchvision import datasets, transforms
from utils import verify_str_arg
from typing import Any, Callable, List, Optional, Union, Tuple
from skimage.transform import rotate, rescale
import random

from retina_transform import foveat_img
from oct2py import octave

octave.addpath(octave.genpath('/private/home/mgahl/logsample'))
#octave.eval('pkg load image')

CSV = namedtuple('CSV', ['header', 'index', 'data'])


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, path, split, target_type, category) -> None:
        super().__init__()

        self.path = path
        self.split = split
        self.root = '/private/home/mgahl/'
        self.base_folder = 'celeba_metadata/'

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if isinstance(category, list):
            self.category = category
        else:
            self.category = [category]

        test_cats_ = ['1757', '2114']
            
        split_map = {
                'train': 0,
                'valid': 1,
                'test': 2,
                'all': None,
                }

        split_ = split_map[verify_str_arg(split.lower(), 'split', ('train', 'valid', 'test', 'all'))]

        cat_map = {
                '>30': 0,
                '30': 1,
                '(20,30)': 2,
                '20': 3,
                'all': 4,
                }

        category_ = [cat_map[x] for x in self.category]


        splits = self._load_csv('new_list_eval_partition.txt')
        categories = self._load_csv('list_category_celeba.txt')
        identity = self._load_csv('identity_CelebA.txt')

        category_ids = self._load_csv('list_category_identities_celeba.txt')

        mask1 = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if len(category_)==1:
            mask2 = slice(None) if category_ is None else (categories.data == category_[0]).squeeze()
        else:
            mask2 = slice(None) if category_ is None else (categories.data == category_[0]).squeeze()
            for i in range(len(category_)-1):
                new_mask = slice(None) if category_ is None else (categories.data == category_[i+1]).squeeze()
                mask2 = torch.logical_or(mask2, new_mask)


        if len(test_cats_) == 1:
            mask3 = slice(None) if test_cats_ is None else (identity.data == int(test_cats_[0])).squeeze()
        elif len(test_cats_) > 1:
            mask3 = slice(None) if test_cats_ is None else (identity.data == int(test_cats_[0])).squeeze()
            for i in range(len(test_cats_)-1):
                new_mask3 = slice(None) if test_cats_ is None else (identity.data == int(test_cats_[i+1])).squeeze()
                mask3 = torch.logical_or(mask3, new_mask3)

        mask = torch.tensor([a and b for a, b in zip(mask1, mask2)])
        mask = torch.tensor([a and b for a, b in zip(mask, mask3)])


        if len(category_)==1:
            id_mask = slice(None) if category_ is None else (category_ids.data == category_[0]).squeeze()
        else:
            id_mask = slice(None) if category_ is None else (category_ids.data == category_[0]).squeeze()
            for i in range(len(category_)-1):
                new_id_mask = slice(None) if category_ is None else (category_ids.data == category_[i+1]).squeeze()
                id_mask = torch.logical_or(id_mask, new_id_mask)


        self.category_ids = [int(x) for x in category_ids.index]
        

        if len(test_cats_)==1:
            id_mask2 = slice(None) if test_cats_ is None else (torch.tensor(self.category_ids) == int(test_cats_[0])).squeeze()
        elif len(test_cats_)>1:
            id_mask2 = slice(None) if test_cats_ is None else (torch.tensor(self.category_ids) == int(test_cats_[0])).squeeze()
            for i in range(len(test_cats_)-1):
                new_id_mask2 = slice(None) if test_cats_ is None else (torch.tensor(self.category_ids) == int(test_cats_[i+1])).squeeze()
                id_mask2 = torch.logical_or(id_mask2, new_id_mask2)

        id_mask = torch.tensor([a and b for a, b in zip(id_mask, id_mask2)])
        self.included_ids = torch.sort(torch.tensor(self.category_ids)[id_mask])

        print ("Number of images in dataset split:", sum(mask1))
        print ("Number of images in category/ies:", sum(mask2))
        print ("Number of images with given identity/ies:", sum(mask3))
        print ("Number of images used:", sum(mask))
        print ("Number of identities used:", sum(id_mask))


        self.filename = splits.index
        self.identity = identity.data[mask]
        self.new_identity = torch.zeros_like(self.identity)
        for i in range(len(self.included_ids[0])):
                self.new_identity = torch.where(self.identity==self.included_ids[0][i], torch.tensor(i), self.new_identity)


    def _load_csv(self, filename, header: Optional[int] = None,) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header+1]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))


    def __getitem__(self, index:int) -> Tuple[Any,Any]:
        X = PIL.Image.open(os.path.join(self.path, self.filename[index]))

        target: any = []
        for t in self.target_type:
            if t == 'identity':
                target.append(self.new_identity[index, 0])
            else:
                raise valueerror('Target type\'{}\' is not recognized.'.format(t))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        else:
            target = None

        return X, target


    def __len__(self) -> int:
        return len(self.new_identity)


    def extra_repr(self) -> str:
        lines = ['Target type: {target_type}', 'Split: {split}']
        return '\n'.join(lines).format(**self.__dict__)



class Collater:
    def __init__(self, log_polar, crop_size, rotations):
        self.log_polar = log_polar
        self.crop_size = crop_size
        self.rotations = rotations

    def __call__(self, samples):
        x, y = zip(*samples)
        norm = transforms.Normalize( mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225) )


        out_x = []
        out_y = []
        for i in range(len(x)):
            crops = self.randomcrop(x[i], 10)
#            crops = transforms.functional.ten_crop(x[i], self.crop_size)
#            crops = [x[i]]
            for j in range(len(crops)):
                for k in range(len(self.rotations)):
                    img = crops[j].rotate(int(self.rotations[k]))
                    img = np.asarray(img, dtype=np.float64)

                    if self.log_polar:
                        img = foveat_img(img, [(img.shape[1] / 2, img.shape[0] / 2)])
                        img = self.logpolar(img, tuple((img.shape[1] / 2, img.shape[0] / 2)))
                        img = Image.fromarray(img)

                    img = np.transpose(img, (2,1,0))

                    img = torch.from_numpy(img)
                    img = norm(img)
                    out_x.append(img)
                    out_y.append(torch.tensor(y[i]))

        return torch.stack(out_x), torch.stack(out_y)

    def randomcrop(self, image1, num_crops):
        width, height = image1.size

        crops = []

        for _ in range(num_crops):
            i = random.randint(0, height - self.crop_size)
            j = random.randint(0, width - self.crop_size)

            cropped_im = transforms.functional.crop(image1, i, j, self.crop_size, self.crop_size)
            crops.append(cropped_im)

        return crops

    def logpolar(self, image1, center):
        log_list = octave.logsample(image1, 5.0, center[0], center[0], center[1], self.out_width, self.out_height)
        octave_logpolar = img_as_ubyte(log_list.astype('B'))

        return np.concatenate((octave_logpolar[100:, :, :], octave_logpolar[:100, :, :]), axis=0)


