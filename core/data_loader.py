"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class SampleDataset(data.Dataset):
    def __init__(self, root, transform=None, trg_domain=0, mode = 'test', dataset_dir='', threshold = None):
        random.seed(123)
        self.root = root
        trg_list = np.load(os.path.join(dataset_dir, '{}_{}.npy'.format(mode, trg_domain)))
        if mode == 'test':
            src_list = np.load(os.path.join(dataset_dir, '{}_{}_src.npy'.format(mode, trg_domain)))
        else: # val during train
            female_list = np.load(os.path.join(dataset_dir, '{}_0.npy'.format(mode)))
            male_list = np.load(os.path.join(dataset_dir, '{}_1.npy'.format(mode)))
            src_list = list(female_list) + list(male_list)
            random.shuffle(src_list)
        src_list = src_list[:threshold]
        trg_list = trg_list[:threshold]
        self.trg_domain = trg_domain
        self.samples = list(zip(list(src_list), list(trg_list)))
        self.transform = transform
        print("data loader target domain: ", trg_domain, ", len: ", len(self.samples))

    def __getitem__(self, index):
        src, trg = self.samples[index]
        img = Image.open(os.path.join(self.root, src)).convert('RGB')
        img2 = Image.open(os.path.join(self.root,trg)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, self.trg_domain

    def __len__(self):
        return len(self.samples)

class SourceDataset(data.Dataset):
    def __init__(self, root, transform = None, mode = 'train',dataset_dir=''):
        female_list = np.load(os.path.join(dataset_dir,'{}_0.npy'.format(mode)))
        male_list = np.load(os.path.join(dataset_dir,'{}_1.npy'.format(mode)))
        img_list = list(female_list) + list(male_list)
        targets = [0 for _ in range(len(female_list))] + [1 for _ in range(len(male_list))]

        self.root = root
        self.samples = img_list
        self.transform = transform
        self.targets = targets
        print("Dataset Len: ", len(self.samples))

    def __getitem__(self, index):
        label = self.targets[index]
        fname = self.samples[index]
        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        img_aug = img
        img_final = self.transform(img_aug)
        return img_final, label

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None,dataset_dir=''):
        self.root = root
        self.samples, self.targets = self._make_dataset(dataset_dir)
        self.transform = transform

    def _make_dataset(self, dataset_dir):
        female_list = np.load(os.path.join(dataset_dir, 'train_0.npy'))
        male_list = np.load(os.path.join(dataset_dir, 'train_1.npy'))
        img_list_dict = {0: list(female_list), 1: list(male_list)}
        fnames, fnames2, labels = [], [], []
        for idx in range(2):
            cls_fnames = img_list_dict[idx]
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        img2 = Image.open(os.path.join(self.root,fname2)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)

class FIDDataset(data.Dataset):
    def __init__(self, root, transform = None, trg_domain = None, threshold = None, mode = 'real', dataset_dir=''):
        if mode == 'real':
            img_list = np.load(os.path.join(dataset_dir, 'train_{}.npy'.format(trg_domain)))
            img_list = list(img_list)
            random.seed(123)
            random.shuffle(img_list)
            img_list = img_list[:threshold] #fid
            root_path = root
        else:
            root_path = os.path.join(root, str(trg_domain))
            img_list = os.listdir(root_path)
        self.root = root_path
        self.samples = img_list
        self.transform = transform
        print("{} dataset Len: ".format(mode), len(self.samples))

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        img_final = self.transform(img)
        return img_final

    def __len__(self):
        return len(self.samples)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4,dataset_dir = ''):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = SourceDataset(root, transform=transform, mode = 'train', dataset_dir=dataset_dir)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform, dataset_dir=dataset_dir)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)

def get_val_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4,dataset_dir = ''):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
    ])

    dataset = SourceDataset(root, transform=transform, mode = 'val', dataset_dir=dataset_dir)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)

def get_fid_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, drop_last=False,
                   mode = 'real', trg_domain = None, fake_len = None,dataset_dir = ''):
    print('Preparing DataLoader for the evaluation phase...')

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = FIDDataset(root, transform=transform, trg_domain=trg_domain, threshold=fake_len, mode = mode, dataset_dir=dataset_dir)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_sample_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, drop_last=False,
                      trg_domain=0, mode = 'test',dataset_dir = '', threshold =None):
    print('Preparing DataLoader for the evaluation phase...')

    height, width = img_size, img_size
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = SampleDataset(root, transform=transform, trg_domain=trg_domain, mode = mode, dataset_dir = dataset_dir, threshold = threshold) # mode = train or eval
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def _fetch_test(self):
        try:
            x, x2, y = next(self.iter_test)
        except (AttributeError, StopIteration):
            self.iter_test = iter(self.loader)
            x, x2, y = next(self.iter_test)
        return x, x2, y

    def __next__(self):
        if self.mode == 'test':
            x, x2, y = self._fetch_test()
            inputs = Munch(src=x, trg= x2, y=y)
        else:
            x, y = self._fetch_inputs()
            if self.mode == 'train':
                x_ref, x_ref2, y_ref = self._fetch_refs()
                z_trg = torch.randn(x.size(0), self.latent_dim)
                z_trg2 = torch.randn(x.size(0), self.latent_dim)
                inputs = Munch(x_src=x, y_src=y,
                               x_ref=x_ref, x_ref2=x_ref2, y_ref=y_ref,
                               z_trg=z_trg, z_trg2=z_trg2)
            elif self.mode == 'val':
                x_ref, y_ref = self._fetch_inputs()
                inputs = Munch(x_src=x, y_src=y,
                               x_ref=x_ref, y_ref=y_ref)
            else:
                raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})