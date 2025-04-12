import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
# import torch

# from tools.utils import mkdir_if_missing, write_json, read_json


class CelebreID(object):
    """ Celeb-reID-light

    Reference:
        Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    # dataset_dir = 'Celeb-reID-light'
    def __init__(self, dataset_root='data',dataset_filename='Celeb-reID', **kwargs):
        self.dataset_dir = osp.join(dataset_root, dataset_filename,'Pad_datasets')
        self.meta_dims = 35
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.mtrain_dir = osp.join(self.dataset_dir, 'mask/train')
        self._check_before_run()
        self.meta_dir='PAR_PETA_0224_0.5.txt'

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> Celeb loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes  # 9021
        self.num_test_clothes = num_test_clothes  # 1821
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        imgdir2attribute = {}
        with open(os.path.join(osp.dirname(self.dataset_dir), self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        images_info = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        camid=0
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            
            mask_path = osp.join(self.mtrain_dir, osp.basename(img_path)[:-4]+ '.png')
            
            clothes = pattern2.search(osp.basename(img_path)).group(1)
            # camid -= 1 # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            description,des_inv,des_cloth=self.cap_gen(imgdir2attribute[img_path])
            
            dataset.append((img_path, pid, clothes_id,camid, imgdir2attribute[img_path],description,des_inv,des_cloth,mask_path))
            camid+=1
            images_info.append({'attributes': imgdir2attribute[img_path]})
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()

        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        imgdir2attribute = {}
        with open(os.path.join(osp.dirname(self.dataset_dir), self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)

        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        images_info_query = []
        images_info_gallery = []
        camid=0
        for img_path in query_img_paths:
            pid, _,_ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            # camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            description,des_inv,des_cloth=self.cap_gen(imgdir2attribute[img_path])
            pid = pid2label[pid]
            
            query_dataset.append((img_path, pid, clothes_id,camid, imgdir2attribute[img_path],description,des_inv,des_cloth,''))
            camid += 1
            images_info_query.append({'attributes': imgdir2attribute[img_path]})

        camid=0
        for img_path in gallery_img_paths:
            pid, _,_ = map(int, pattern1.search(img_path).groups())#, camid
            clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            # camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            description,des_inv,des_cloth=self.cap_gen(imgdir2attribute[img_path])
            pid = pid2label[pid]
            
            gallery_dataset.append((img_path, pid, clothes_id,camid, imgdir2attribute[img_path],description,des_inv,des_cloth,''))
            camid+=1
            images_info_gallery.append({'attributes': imgdir2attribute[img_path]})
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
    
    def cap_gen(self,attrlb):
        description=''
        des_inv=''
        
        if attrlb[34]==1:
            description +="The "+ 'man' 
        else:
            description +="The "+ 'woman' 
            
        # des_cloth="A pedestrian with "  
            
        age=[" under the age of 30", " between the ages of 30 and 45",
                " between the ages of 45 and 60", " over the age of 60"]
        ageind=0
        for i in range(30,34):
            if  attrlb[i]==1:
                if ageind>0:
                    description += ' or'
                    # des_cloth += ' or'
                description += age[i-30]
                # des_cloth += age[i-30]
                ageind+=1
        if ageind==0:
            assert(" unknown years old")
            
        des_inv=description
        
        des_cloth = 'A pedestrain with '   
        description +=", with " 
        # des_cloth +=", with "   
        if attrlb[4]==1:
            description += "long "
            des_cloth += "long "
        else:
            description += "short "
            des_cloth += "short "
        
        description +="hair"
        # des_inv += "hair."
        des_cloth +="hair, "
        
        
        
        carry=["backpack", "other types of attachments",  "messenger bag",  "no attachments","plastic bag"]
        carryind=0
        for i in range(25,30):
            if  attrlb[i]==1:
                description += ", "
                description += carry[i-25]
                if carryind > 1:
                    des_cloth += ", "
                des_cloth += carry[i-25]
                carryind+=1
        if carryind==0:
            description +=", unknown carrying"
            # description +=carry[3]
            des_cloth += "unknown carrying"
            
        
        headwear=[ "hat", "muffler", "no headwear","sunglasses"]
        headind=0
        for i in range(4):
            if  attrlb[i]==1:
                
                description += ", "
                des_cloth += ", "
                description += headwear[i]
                des_cloth += headwear[i]
                headind+=1
        if headind==0:
            description +="unknown headwear"
            des_cloth += "unknown headwear"
        description +=", "
        des_cloth += ", "
            
        if attrlb[5]==1:
            description += "casual upper wear"
            des_cloth += "casual upper wear"
        elif attrlb[6]==1:
            description += "formal upper wear"
            des_cloth += "formal upper wear"
        else:
            description += "uunknown style's upper wear"
            des_cloth += "unknown style's upper wear"
        
        upercloth=["jacket", "logo upper wear", "plaid upper wear", "short sleeves",  "thin stripes upper wear","t-shirt", "other upper wear","vneck upper wear"]
        uperwear=0
        for i in range(7,15):
            if  attrlb[i]==1:
                description +=', '
                description += upercloth[i-7]
                des_cloth +=', '
                des_cloth += upercloth[i-7] 
                uperwear+=1
        if uperwear==0:
            description += ", unknown upper wear" 
            des_cloth += ", unknown upper wear" 
        
        description += ", " 
        des_cloth += ", " 
        #lowbody    
        if attrlb[15]==1:
            description += "casual lower wear"
            des_cloth += "casual lower wear"
        elif attrlb[16]==1:
            description += "formal lower wear"
            des_cloth += "formal lower wear"
        else:
            description += "unknown style's lower wear" 
            des_cloth += "unknown style's lower wear"
        
                
        lowercloth=["jeans","shorts","short skirt","trousers"]
        lowerwear=0
        for i in range(17,21):
            if  attrlb[i]==1:
                description += ", " 
                des_cloth += ", " 
                description += lowercloth[i-17]
                des_cloth += lowercloth[i-17]
                lowerwear+=1
        if lowerwear==0:
            description +=", unknown lower wear" 
            des_cloth += ", unknown lower wear" 
        # description += ", and "
        # des_cloth += ", and "
        #footwear
        footwear=["leather shoes","sandals",'other types of shoes', "sneaker"]#["leatherShoes","sandals", "sneaker",'shoes']
        footind=0
        for i in range(21,25):
            if  attrlb[i]==1:
                description += ", " 
                des_cloth += ", " 
                description += footwear[i-21]
                des_cloth += footwear[i-21]
                footind+=1
        if footind==0:
            description +="unknown shoes"
            des_cloth +="unknown shoes" 
            
        description += "."
        des_cloth +="." 
        description = pre_caption(description,77)
        des_inv = pre_caption(des_inv,77)
        des_cloth = pre_caption(des_cloth,77)
        return description,des_inv,des_cloth
    

    
def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption    

if __name__ =='__main__':
    dataset=Celeb_light('/data/Data/ReIDData')
    print(dataset.num_train_clothes)
    print(dataset.num_test_clothes)