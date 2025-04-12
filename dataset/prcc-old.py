# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import torch
import re
import numpy as np
import os.path as osp

from .bases import BaseImageDataset

from args import argument_parser
# global variables
parser = argument_parser()
args = parser.parse_args()

class Prcc(BaseImageDataset):
    """
    PRCC
    Reference:
    Person Re-identification by Contour Sketch under Moderate Clothing Change.
    TPAMI-2019

    Dataset statistics:
    # identities: 221, with 3 camera views.
    # images: 150IDs (train) + 71IDs (test)

    Dataset statistics: (A--->C, cross-clothes settings)
      ----------------------------------------
      subset   | # ids | # cloth_ids | # images | # cameras
      ----------------------------------------
      train    |   150 |     3 |    17896 |         3
      query    |    71 |     1 |     3384 |         1
      gallery  |    71 |     1 |     3543 |         1
      ----------------------------------------

    Two test settings:
    parser.add_argument('--cross-clothes', action='store_true',
                        help="the person matching between Camera views A and C was cross-clothes matching")
    parser.add_argument('--same-clothes', action='store_true',
                        help="the person matching between Camera views A and B was performed without clothing changes")
    """
    dataset_dir = 'prcc/Pad_datasets1/rgb' # could change to sketch/contour folder

    def __init__(self, root='/home/jinx/data', verbose=True, **kwargs):
        super(Prcc, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.validation_dir = osp.join(self.dataset_dir, 'val')
        self.probe_gallery_dir = osp.join(self.dataset_dir, 'test')
        
        self.meta_dir='PAR_PETA_35.txt'
        
        self._check_before_run()
        self.meta_dims=35
        imgdir2attribute = {}
        
        with open(osp.join(root, 'prcc',self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present) 

        train,pid2clothes = self._process_dir(self.train_dir, imgdir2attribute, relabel=True)
        query, gallery = self._process_test_dir(self.probe_gallery_dir,imgdir2attribute, relabel=False)

        if verbose:
            print("=> PRCC dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.pid2clothes = pid2clothes
        
        
                
        self.num_train_pids, self.num_train_cloth_ids, self.num_train_imgs, self.num_train_cams, self.num_train_attr = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_cloth_ids, self.num_query_imgs, self.num_query_cams, self.num_query_attr = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_cloth_ids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_attr = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.probe_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.probe_gallery_dir))


    def _process_dir(self, dir_path,imgdir2attribute, relabel=False):

        # Load from train
        pid_dirs_path = glob.glob(osp.join(dir_path, '*'))

        dataset = []
        pid_container = set()
        clothes_container = set()
        camid_mapper = {'A': 1, 'B': 2, 'C': 3}
        for pdir in pid_dirs_path:
            img_paths = glob.glob(osp.join(pdir, '*.jp*'))
            for img_dir in img_paths:
                pid = int(osp.basename(pdir))
                pid_container.add(pid)
                cam=osp.basename(img_dir)[0]
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        
        pid2clothes = np.zeros((num_pids, num_clothes))
        label_all=[]
        for pdir in pid_dirs_path:
            img_paths = glob.glob(osp.join(pdir, '*.jp*'))
            for img_dir in img_paths:
                pid = int(osp.basename(pdir))
                cam = osp.basename(img_dir)[0]
                camid = camid_mapper[cam]  
                # cloth_id = camid
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir)+osp.basename(img_dir)[0]]
                
                label_all.append(imgdir2attribute[img_dir])
                #gen description
                description=''
                des_inv=''
                
                if imgdir2attribute[img_dir][34]==1:
                    description +="The "+ 'man' 
                else:
                    description +="The "+ 'woman' 
                    
                age=[" under the age of 30 ", " between the ages of 30 and 45 ",
                     " between the ages of 45 and 60 ", " over the age of 60 "]
                ageind=0
                for i in range(30,34):
                    if  imgdir2attribute[img_dir][i]==1:
                        description += age[i-30]
                        ageind+=1
                if ageind==0:
                    assert("unknown age！")
                    
                description +="with "     
                if imgdir2attribute[img_dir][4]==1:
                    description += "long "
                else:
                    description += "short "
                des_inv=description
                description +="hair"
                des_inv += "hair."
                
                des_cloth="A pedestrian with "
                
                carry=["backpack", "other types of attachments",  "messenger bag",  "no attachments","plastic bag"]
                carryind=0
                for i in range(25,30):
                    if  imgdir2attribute[img_dir][i]==1:
                        description += " and "
                        description += carry[i-25]
                        if carryind > 1:
                            des_cloth += " and "
                        des_cloth += carry[i-25]
                        carryind+=1
                if carryind==0:
                    description +=" and unknown carrying"
                    # description +=carry[3]
                    des_cloth += "and unknown carrying"
                    
                description += " wears "
                des_cloth += " wears "
                headwear=[ "hat", "muffler", "no headwear","sunglasses"]
                headind=0
                for i in range(4):
                    if  imgdir2attribute[img_dir][i]==1:
                        description += headwear[i]
                        des_cloth += headwear[i]
                        headind+=1
                if headind==0:
                    description +="unknown headwear"
                    des_cloth += "unknown headwear"
                description +=", "
                des_cloth += ", "
                    
                if imgdir2attribute[img_dir][5]==1:
                    description += "casual "
                    des_cloth += "casual "
                elif imgdir2attribute[img_dir][6]==1:
                    description += "formal "
                    des_cloth += "formal "
                else:
                    description += "unknown upperstyle's "
                    des_cloth += "unknown upperstyle's "
                
                upercloth=["jacket", "logo", "plaid", "shortSleeve",  "thinStripes","t-shirt", "other upper cloth","vneck"]
                uperwear=0
                for i in range(7,15):
                    if  imgdir2attribute[img_dir][i]==1:
                        description += upercloth[i-7]
                        description +=', '
                        des_cloth += upercloth[i-7] 
                        des_cloth +=', '
                        uperwear+=1
                if uperwear==0:
                    description += upercloth[6] 
                    des_cloth += upercloth[6] 
                
                #lowbody    
                if imgdir2attribute[img_dir][15]==1:
                    description += "casual "
                    des_cloth += "casual "
                elif imgdir2attribute[img_dir][16]==1:
                    description += "formal "
                    des_cloth += "formal "
                else:
                    description += "unknown lowerstyle's " 
                    des_cloth += "unknown lowerstyle's "
                     
                lowercloth=["jeans","shorts","short skirt","trousers"]
                lowerwear=0
                for i in range(17,21):
                    if  imgdir2attribute[img_dir][i]==1:
                        description += lowercloth[i-17]
                        des_cloth += lowercloth[i-17]
                        lowerwear+=1
                if lowerwear==0:
                    description +="unknown lowerwear" 
                    des_cloth += "unknown lowerwear" 
                description += ", and "
                des_cloth += ", and "
                #footwear
                footwear=["leatherShoes","sandals", "sneaker",'shoes']
                footind=0
                for i in range(21,25):
                    if  imgdir2attribute[img_dir][i]==1:
                        description += footwear[i-21]
                        des_cloth += footwear[i-21]
                        footind+=1
                if footind==0:
                    description +="shoes"
                    des_cloth +="shoes" 
                    
                description += "."
                des_cloth +="." 
                
        
                    
                dataset.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))
                pid2clothes[pid, clothes_id] = 1

        return dataset,pid2clothes

    def _process_test_dir(self, dir_path, imgdir2attribute, relabel=False):

        camid_dirs_path = glob.glob(osp.join(dir_path, '*'))

        query = []
        gallery = []
        camid_mapper = {'A': 1, 'B': 2, 'C': 3}

        pid_container = set()
        for pdir in glob.glob(osp.join(dir_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        
        for camid_dir_path in camid_dirs_path:
            pid_dir_paths = glob.glob(osp.join(camid_dir_path, '*'))
            for pid_dir_path in pid_dir_paths:
                pid = int(osp.basename(pid_dir_path))
                img_paths = glob.glob(osp.join(pid_dir_path, '*'))
                for img_dir in img_paths:
                    #gen description
                    description=''
                    des_inv=''
                    
                    if imgdir2attribute[img_dir][34]==1:
                        description +="The "+ 'man' 
                    else:
                        description +="The "+ 'woman' 
                        
                    age=[" under the age of 30 ", " between the ages of 30 and 45 ",
                        " between the ages of 45 and 60 ", " over the age of 60 "]
                    ageind=0
                    for i in range(30,34):
                        if  imgdir2attribute[img_dir][i]==1:
                            description += age[i-30]
                            ageind+=1
                    if ageind==0:
                        assert("unknown age！")
                        
                    description +="with "     
                    if imgdir2attribute[img_dir][4]==1:
                        description += "long "
                    else:
                        description += "short "
                    des_inv=description
                    description +="hair"
                    des_inv += "hair."
                    
                    des_cloth="A pedestrian with "
                    
                    carry=["backpack", "other types of attachments",  "messenger bag",  "no attachments","plastic bag"]
                    carryind=0
                    for i in range(25,30):
                        if  imgdir2attribute[img_dir][i]==1:
                            description += " and "
                            description += carry[i-25]
                            if carryind > 1:
                                des_cloth += " and "
                            des_cloth += carry[i-25]
                            carryind+=1
                    if carryind==0:
                        description +=" and unknown carrying"
                        # description +=carry[3]
                        des_cloth += "and unknown carrying"
                        
                    description += " wears "
                    des_cloth += " wears "
                    headwear=[ "hat", "muffler", "no headwear","sunglasses"]
                    headind=0
                    for i in range(4):
                        if  imgdir2attribute[img_dir][i]==1:
                            description += headwear[i]
                            des_cloth += headwear[i]
                            headind+=1
                    if headind==0:
                        description +="unknown headwear"
                        des_cloth += "unknown headwear"
                    description +=", "
                    des_cloth += ", "
                        
                    if imgdir2attribute[img_dir][5]==1:
                        description += "casual "
                        des_cloth += "casual "
                    elif imgdir2attribute[img_dir][6]==1:
                        description += "formal "
                        des_cloth += "formal "
                    else:
                        description += "unknown upperstyle's "
                        des_cloth += "unknown upperstyle's "
                    
                    upercloth=["jacket", "logo", "plaid", "shortSleeve",  "thinStripes","t-shirt", "other upper cloth","vneck"]
                    uperwear=0
                    for i in range(7,15):
                        if  imgdir2attribute[img_dir][i]==1:
                            description += upercloth[i-7]
                            description +=', '
                            des_cloth += upercloth[i-7] 
                            description +=', '
                            uperwear+=1
                    if uperwear==0:
                        description += upercloth[6] 
                        des_cloth += upercloth[6] 
                    #lowbody    
                    if imgdir2attribute[img_dir][15]==1:
                        description += "casual "
                        des_cloth += "casual "
                    elif imgdir2attribute[img_dir][16]==1:
                        description += "formal "
                        des_cloth += "formal "
                    else:
                        description += "unknown lowerstyle's " 
                        des_cloth += "unknown lowerstyle's "
                        
                    lowercloth=["jeans","shorts","short skirt","trousers"]
                    lowerwear=0
                    for i in range(17,21):
                        if  imgdir2attribute[img_dir][i]==1:
                            description += lowercloth[i-17]
                            des_cloth += lowercloth[i-17]
                            lowerwear+=1
                    if lowerwear==0:
                        description +="unknown lowerwear" 
                        des_cloth += "unknown lowerwear" 
                    description += ", and "
                    des_cloth += ", and "
                    #footwear
                    footwear=["leatherShoes","sandals", "sneaker",'shoes']
                    footind=0
                    for i in range(21,25):
                        if  imgdir2attribute[img_dir][i]==1:
                            description += footwear[i-21]
                            des_cloth += footwear[i-21]
                            footind+=1
                    if footind==0:
                        description +="shoes"
                        des_cloth +="shoes" 
                        
                    description += "."
                    des_cloth +="."
                
                    camid = camid_mapper[osp.basename(camid_dir_path)]
                    camid -= 1  # index starts from 0
                    if camid == 0:
                        # cloth_id = camid
                        clothes_id = pid2label[pid] * 2
                        gallery.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))
                    else:
                        if args.cross_clothes and camid == 2:
                            # cloth_id = camid
                            clothes_id = pid2label[pid] * 2 + 1
                            query.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))#[1:]
                        elif args.same_clothes and camid == 1:
                            # cloth_id = camid
                            clothes_id = pid2label[pid] * 2 
                            query.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))
                    # if camid == 0:
                    #     # cloth_id = camid
                    #     clothes_id = pid2label[pid] * 2
                    #     query.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))
                    # else:
                    #     if args.cross_clothes and camid == 2:
                    #         # cloth_id = camid
                    #         clothes_id = pid2label[pid] * 2
                    #         gallery.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))#[1:]
                    #     elif args.same_clothes and camid == 1:
                    #         # cloth_id = camid
                    #         clothes_id = pid2label[pid] * 2 + 1
                    #         gallery.append((img_dir, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth))

        return query, gallery
