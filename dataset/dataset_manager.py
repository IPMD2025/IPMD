from __future__ import print_function, absolute_import

from dataset.LTCC import LTCC
from dataset.CelebreID import CelebreID
from dataset.DeepChange import DeepChange
from dataset.LaST import LaST
from dataset.vc_clothe import VC_clothe
from dataset.Celeb_light import Celeb_light


__img_factory = {
    'ltcc': LTCC,
    'celeb': CelebreID,
    'celeb-light':Celeb_light,
    'deepchange': DeepChange,
    'last': LaST,
    'vc-clothes':VC_clothe
}


def get_dataset(args):
    name = args.dataset
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))

    dataset = __img_factory[name](dataset_root=args.dataset_root, dataset_filename=args.dataset_filename)
    return dataset