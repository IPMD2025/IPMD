"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import print_function, absolute_import

from utils.util import read_image
from data_process import samplers, transform

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_process.mask import read_person_mask
from torchvision import transforms as T
import torch


class ImageClothDataset_cc(Dataset):
    def __init__(self, dataset, transform=None,transform_s=None,transform_v=None):
        self.dataset = dataset#[:2000]
        self.transform = transform
        self.tf_shape = transform_s
        self.tf_value = transform_v
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, clothid, camid, attrlabel,des,des_inv,des_cloth,mask_path = self.dataset[index]
        img = read_image(img_path)
        
        if (self.tf_shape is not None) and (self.tf_value is not None):
            # mask = read_image(mask_path)
            # if self.transform is not None:
            #     img = self.transform(img)
            #     mask = self.transform(mask)
            img = self.to_tensor(img)
            mask = torch.tensor(read_person_mask(mask_path)).unsqueeze(0).repeat(3,1,1)
            sem=img*mask + (1-mask)*255
            img_sem = torch.cat([img,mask], dim = 0)
            # shape transform
            img_sem = self.tf_shape(img_sem)
            img = img_sem[0:3]
            sem = img_sem[3:]
            # value transform
            img = self.tf_value(img)
            sem = self.tf_value(sem)
            return img, pid, clothid, camid,attrlabel,des,des_inv,des_cloth,sem
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, pid, clothid, camid,attrlabel,des,des_inv,des_cloth


def get_prcc_dataset_loader(dataset, args, use_gpu=True):
    transform_train, transform_test,transform_shape,transform_value = transform.get_transform(args)

    sampler = samplers.RandomIdentitySampler_cc(dataset.train, batch_size=args.train_batch,
                                                num_instances=args.num_instances)

    pin_memory = use_gpu
    train_loader = DataLoader(
        ImageClothDataset_cc(dataset.train, transform=transform_train,transform_s=transform_shape,transform_v=transform_value),
        sampler=sampler,
        batch_size=args.train_batch, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )

    query_sc_loader = DataLoader(
        ImageClothDataset_cc(dataset.query_cloth_unchanged, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    query_cc_loader = DataLoader(
        ImageClothDataset_cc(dataset.query_cloth_changed, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    gallery_loader = DataLoader(
        ImageClothDataset_cc(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return train_loader, query_sc_loader, query_cc_loader, gallery_loader


def get_cc_dataset_loader(dataset, args, use_gpu=True):
    transform_train, transform_test,transform_shape,transform_value = transform.get_transform(args)

    sampler = samplers.RandomIdentitySampler_cc(dataset.train, batch_size=args.train_batch,
                                                num_instances=args.num_instances)

    pin_memory = use_gpu
    train_loader = DataLoader(
        ImageClothDataset_cc(dataset.train, transform=transform_train,transform_s=transform_shape,transform_v=transform_value),
        sampler=sampler,
        batch_size=args.train_batch, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )

    query_loader = DataLoader(
        ImageClothDataset_cc(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    gallery_loader = DataLoader(
        ImageClothDataset_cc(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return train_loader, query_loader, gallery_loader