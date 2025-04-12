import glob
import random
import numpy as np
import os.path as osp
import torch
import re
import numpy as np


class PRCC(object):
    """ PRCC

    Reference:
        Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.

    URL: https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view
    """
    dataset_dir = 'prcc/Pad_datasets1/rgb'
    def __init__(self, dataset_root='data', dataset_filename='prcc', verbose=True, **kwargs):
        self.dataset_dir = osp.join(dataset_root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.mtrain_dir = osp.join(dataset_root, 'prcc/Pad_datasets1/mask/train')
        self.mval_dir = osp.join(dataset_root, 'prcc/Pad_datasets1/mask/val')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.meta_dir='PAR_PETA_1220_0.5.txt'
        self._check_before_run()
        self.meta_dims=35
        imgdir2attribute = {}
        
        with open(osp.join(dataset_root, 'prcc',self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present) 


        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir, imgdir2attribute)
        val, num_val_pids, num_val_imgs, num_val_clothes, _ = \
            self._process_dir_train(self.val_dir,imgdir2attribute,)

        query_same, query_diff, gallery, num_test_pids, \
        num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
        num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir,imgdir2attribute,)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        if verbose:
            print("=> PRCC loaded")
            print("Dataset statistics:")
            print("  --------------------------------------------")
            print("  subset      | # ids | # images | # clothes")
            print("  --------------------------------------------")
            print("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
            print("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_clothes))
            # print("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
            print("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
            print("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
            print("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
            print("  --------------------------------------------")
            print("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  --------------------------------------------")

        self.train = train
        self.val = val
        self.query_cloth_unchanged = query_same
        self.query_cloth_changed = query_diff
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.gallery_idx = gallery_idx

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir_train(self, dir_path,imgdir2attribute):
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]  # 'A' or 'B' or 'C'
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir) + osp.basename(img_dir)[0])
        pid_container = sorted(list(pid_container))
        clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]  # 'A' or 'B' or 'C'
                mask_path = osp.join(self.mtrain_dir, osp.basename(pdir),osp.basename(img_dir)[:-4]+ '.png')
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir) + osp.basename(img_dir)[0]]
                
                pid2clothes[label, clothes_id] = 1
                
                ########################
                description=''
                des_inv=''
                
                if imgdir2attribute[img_dir][34]==1:
                    description +="The "+ 'man' 
                else:
                    description +="The "+ 'woman' 
                    
                age=[" under the age of 30", " between the ages of 30 and 45",
                     " between the ages of 45 and 60", " over the age of 60"]
                ageind=0
                for i in range(30,34):
                    if  imgdir2attribute[img_dir][i]==1:
                        if ageind>0:
                            description += ' or'
                        description += age[i-30]
                        ageind+=1
                if ageind==0:
                    assert(" unknown years old")
                des_inv=description
                    
                des_cloth="A pedestrian with "   
                description +=", with "     
                if imgdir2attribute[img_dir][4]==1:
                    description += "long "
                    des_cloth += "long "
                else:
                    description += "short "
                    des_cloth += "short "
                
                description +="hair"
                des_cloth += "hair, "
                # des_inv += "hair."
                
                
                 
                carry=["backpack", "other types of attachments",  "messenger bag",  "no attachments","plastic bag"]
                carryind=0
                for i in range(25,30):
                    if  imgdir2attribute[img_dir][i]==1:
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
                    if  imgdir2attribute[img_dir][i]==1:
                        
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
                    
                if imgdir2attribute[img_dir][5]==1:
                    description += "casual upper wear"
                    des_cloth += "casual upper wear"
                elif imgdir2attribute[img_dir][6]==1:
                    description += "formal upper wear"
                    des_cloth += "formal upper wear"
                else:
                    description += "uunknown style's upper wear"
                    des_cloth += "unknown style's upper wear"
                
                upercloth=["jacket", "logo upper wear", "plaid upper wear", "short sleeves",  "thin stripes upper wear","t-shirt", "other upper wear","vneck upper wear"]
                uperwear=0
                for i in range(7,15):
                    if  imgdir2attribute[img_dir][i]==1:
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
                if imgdir2attribute[img_dir][15]==1:
                    description += "casual lower wear"
                    des_cloth += "casual lower wear"
                elif imgdir2attribute[img_dir][16]==1:
                    description += "formal lower wear"
                    des_cloth += "formal lower wear"
                else:
                    description += "unknown style's lower wear" 
                    des_cloth += "unknown style's lower wear"
                
                     
                lowercloth=["jeans","shorts","short skirt","trousers"]
                lowerwear=0
                for i in range(17,21):
                    if  imgdir2attribute[img_dir][i]==1:
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
                    if  imgdir2attribute[img_dir][i]==1:
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
                # dataset.append((img_dir, label, clothes_id, camid))
                dataset.append((img_dir, label, clothes_id, camid,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth,mask_path))
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, test_path,imgdir2attribute):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    # pid = pid2label[pid]
                    #######################
                    description=''
                    des_inv=''
                    
                    if imgdir2attribute[img_dir][34]==1:
                        description +="The "+ 'man' 
                    else:
                        description +="The "+ 'woman' 
                        
                    age=[" under the age of 30", " between the ages of 30 and 45",
                        " between the ages of 45 and 60", " over the age of 60"]
                    ageind=0
                    for i in range(30,34):
                        if  imgdir2attribute[img_dir][i]==1:
                            if ageind>0:
                                description += ' or'
                            description += age[i-30]
                            ageind+=1
                    if ageind==0:
                        assert(" unknown years old")
                        
                    description +=", with "     
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
                        if  imgdir2attribute[img_dir][i]==1:
                            
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
                        
                    if imgdir2attribute[img_dir][5]==1:
                        description += "casual upper wear"
                        des_cloth += "casual upper wear"
                    elif imgdir2attribute[img_dir][6]==1:
                        description += "formal upper wear"
                        des_cloth += "formal upper wear"
                    else:
                        description += "uunknown style's upper wear"
                        des_cloth += "unknown style's upper wear"
                    
                    upercloth=["jacket", "logo upper wear", "plaid upper wear", "short sleeves",  "thin stripes upper wear","t-shirt", "other upper wear","vneck upper wear"]
                    uperwear=0
                    for i in range(7,15):
                        if  imgdir2attribute[img_dir][i]==1:
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
                    if imgdir2attribute[img_dir][15]==1:
                        description += "casual lower wear"
                        des_cloth += "casual lower wear"
                    elif imgdir2attribute[img_dir][16]==1:
                        description += "formal lower wear"
                        des_cloth += "formal lower wear"
                    else:
                        description += "unknown style's lower wear" 
                        des_cloth += "unknown style's lower wear"
                    
                        
                    lowercloth=["jeans","shorts","short skirt","trousers"]
                    lowerwear=0
                    for i in range(17,21):
                        if  imgdir2attribute[img_dir][i]==1:
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
                        if  imgdir2attribute[img_dir][i]==1:
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
                            
                    ########################
                    camid = cam2label[cam]
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        gallery_dataset.append((img_dir, pid2label[pid], clothes_id, camid, torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth,''))
                    elif cam == 'B':
                        clothes_id = pid2label[pid] * 2
                        query_dataset_same_clothes.append((img_dir, pid2label[pid], clothes_id, camid ,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth,''))
                    else:
                        clothes_id = pid2label[pid] * 2 + 1
                        query_dataset_diff_clothes.append((img_dir, pid2label[pid], clothes_id, camid ,torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth,''))
                    
        pid2imgidx = {}
        for idx, (img_dir, pid, camid, clothes_id,_,_,_,_,_) in enumerate(gallery_dataset):
            if pid not in pid2imgidx:
                pid2imgidx[pid] = []
            pid2imgidx[pid].append(idx)

        # get 10 gallery index to perform single-shot test
        gallery_idx = {}
        random.seed(3)
        for idx in range(0, 10):
            gallery_idx[idx] = []
            for pid in pid2imgidx:
                gallery_idx[idx].append(random.choice(pid2imgidx[pid]))

        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
               num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
               num_clothes, gallery_idx

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