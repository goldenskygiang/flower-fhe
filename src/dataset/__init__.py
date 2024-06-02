import pandas as pd
import os

import torch
import torchvision.transforms as tvtf
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10

from PIL import Image
from typing import Tuple

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

def encode_label(labels, classes=VOC_CLASSES):
    '''
    Takes a list of label names as input, encodes them into a binary tensor:
        - Same length as len(VOC_CLASSES)
        - Each element represents the presence (1) or absence (0) of a specific class

    '''
    target = torch.zeros(len(classes))
    for l in labels:
        try:
            idx = classes.index(l)
            target[idx] = 1 # 1 if label is present
        except Exception as e:
            # print(l)
            # print(labels)
            continue

    return target

class PascalVOCDataset(Dataset):
    def __init__(self, data_path, csv_file, transform=None):
        self.data_path = data_path
        self.csv_file = csv_file
        self.transform = transform

        # load CSV file containing annotations
        self.annotations = pd.read_csv(os.path.join(data_path, csv_file)) # N * 3

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        # get image filename and label at specified index:
        img_filename = self.annotations.iloc[index, 0]
        label = self.annotations.iloc[index, 1].replace(" ", "")

        label = label.split(',')

        # load image
        img_path = os.path.join(self.data_path, 'JPEGImages', img_filename)
        #img = Image.open(img_path).convert('RGB')

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Handle the exception when the image file is not found
            print(f"Image file not found: {img_path}")
            # Create a default black image
            img = Image.new('RGB', (300, 300), (int(0.5 * 255), int(0.5 * 255), int(0.5 * 255)))
            # Return default label with all classes not present
            label = encode_label([])  # Empty list for all classes not present
            #return img, label

        # apply transformations
        if self.transform:
            img = self.transform(img)

        return img, encode_label(label)


class CifarDataset(Dataset):
    def __init__(self, data_path, ver: str='10', train=True, transform=None):
        self.cifar = None
        if ver == '10':
            self.cifar = CIFAR10(root=data_path, train=train, download=True, transform=transform)
        else:
            self.cifar = CIFAR100(root=data_path, train=train, download=True, transform=transform)

        self.transform = transform

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int] :
        img = self.cifar.data[index]
        label = self.cifar.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, label


def prep_data_cifar(data_path, ver: str='10', val_split=0.2):
    # tf = tvtf.Compose([
    #     tvtf.ToTensor(),
    #     tvtf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    normalize = tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    train_transform = tvtf.Compose([
    tvtf.ToTensor(),
    tvtf.Resize(256),        # Resize to 256x256 first
    tvtf.RandomCrop(224),     # Randomly crop to 224x224
    tvtf.RandomHorizontalFlip(),
    normalize,  # Use ImageNet normalization
    ])

    val_transform = tvtf.Compose([  # For validation and testing
    tvtf.ToTensor(),
    tvtf.Resize(224),  # Resize directly to 224x224
    normalize,
    ])

    full_train = CifarDataset(data_path, ver, train=True, transform=train_transform)

    # Split the dataset into train, val
    total_size = len(full_train)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    ds_train, ds_val = torch.utils.data.random_split(full_train, [train_size, val_size])

    ds_test = CIFAR10(data_path, train=False, transform=val_transform)

    return ds_train, ds_val, ds_test


def prep_data_pascal(data_path):
    # Transform
    mean = [.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = tvtf.Compose([
        tvtf.Resize((300, 300)), tvtf.RandomCrop((256, 256)),
        tvtf.RandomHorizontalFlip(p=0.2),
        tvtf.ToTensor(), tvtf.Normalize(mean=mean, std=std)
    ])

    eval_test_transform = tvtf.Compose([
        tvtf.Resize((300, 300)), tvtf.CenterCrop((256, 256)),
        tvtf.RandomHorizontalFlip(p=0.25),
        tvtf.ToTensor(), tvtf.Normalize(mean=mean, std=std)
    ])

    ds_train = PascalVOCDataset(data_path, 'train.csv', transform=train_transform)
    ds_valid = PascalVOCDataset(data_path, 'valid.csv', transform=train_transform)
    ds_test = PascalVOCDataset(data_path, 'test.csv', transform=eval_test_transform)

    return ds_train, ds_valid, ds_test


def prep_data_decentralized(
        ds_name: str, data_path: str, num_partitions: int,
        batch_size: int=128, num_workers: int=0, **kwargs):
    '''
    Partitions the training / validation set into N disjoint subsets,
    each of which will become the local dataset of the
    simulated client. Test set is left intact and used later for
    global model evaluation.
    '''
    # get train, test
    if ds_name == 'pascal':
        ds_train, ds_val, ds_test = prep_data_pascal(data_path)
    else:
        cifar_ver = kwargs['cifar_ver'] if 'cifar_ver' in kwargs.keys() else '10'
        val_split = kwargs['cifar_val_split'] if 'cifar_val_split' in kwargs.keys() else 0.15
        ds_train, ds_val, ds_test = prep_data_cifar(data_path, cifar_ver, val_split)

    # split trainset, valset into n partitions
    num_images_train = len(ds_train) // num_partitions
    partition_len_train = [num_images_train] * num_partitions
    partition_len_train[-1] = len(ds_train) - sum(partition_len_train[:-1]) # last subset may not have enough samples

    trainsets = random_split(ds_train, partition_len_train, torch.Generator().manual_seed(2023))

    num_images_val = len(ds_val) // num_partitions
    partition_len_val = [num_images_val] * num_partitions
    partition_len_val[-1] = len(ds_val) - sum(partition_len_val[:-1]) # last subset may not have enough samples
    valsets = random_split(ds_val, partition_len_val, torch.Generator().manual_seed(2023))

    # create dataloader
    dl_trains = []
    dl_vals = []

    for trainset_ in trainsets:
        # num_total = len(trainset_)
        dl_trains.append(DataLoader(trainset_, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True))

    for valset_ in valsets:
        num_total = len(valset_)
        dl_vals.append(DataLoader(valset_, batch_size=batch_size, num_workers=num_workers, drop_last=True))

    # test
    dl_test = DataLoader(ds_test, batch_size=batch_size, drop_last=True)

    return dl_trains, dl_vals, dl_test