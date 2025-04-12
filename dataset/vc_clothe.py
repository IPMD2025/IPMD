import glob
import re
import torch
import os.path as osp

from dataset.base_image_dataset import BaseImageDataset


class VC_clothe(BaseImageDataset):
    """
       VC_clothes dataset
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'VC-Clothes/Pad_datasets'

    def __init__(self, dataset_root='data',dataset_filename='VC-Clothes', verbose=True, **kwargs):
        super(VC_clothe, self).__init__()
        self.dataset_dir = osp.join(dataset_root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.mtrain_dir = osp.join(self.dataset_dir, 'mask/train')
        self.mquery_dir = osp.join(self.dataset_dir, 'mask/query')
        self.mgallery_dir = osp.join(self.dataset_dir, 'mask/test')
        self.meta_dir='PAR_PETA_0219_0.5.txt'

        self._check_before_run()
        self.meta_dims=35
        imgdir2attribute = {}
        
        with open(osp.join(dataset_root, dataset_filename,self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present) 
                
        train,num_train_pids, num_train_imgs, num_train_clothes = self._process_dir(self.train_dir,imgdir2attribute,relabel=True)
        query,num_query_pids, num_query_imgs, num_query_clothes = self._process_dir(self.query_dir,imgdir2attribute, relabel=False)
        gallery,num_gallery_pids, num_gallery_imgs, num_gallery_clothes = self._process_dir(self.gallery_dir,imgdir2attribute, relabel=False)
        
        # num_total_pids = num_train_pids + num_test_pids
        # num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # # num_test_imgs = num_query_imgs + num_gallery_imgs
        # num_total_clothes = num_train_clothes + num_test_clothes

        # if verbose:
        #     print("=> VC_clothes dataset is loaded")
        #     self.print_dataset_statistics(train, query, gallery)
        if verbose:
            print("=> LTCC loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
            # print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
            print("  query    | {:5d} | {:8d} |".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d} |".format(num_gallery_pids, num_gallery_imgs))
            print("  ----------------------------------------")
            # print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  ----------------------------------------")
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

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

    def _process_dir(self, dir_path,imgdir2attribute, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jp*'))
        pattern = re.compile(r'([\d]+)-([\d]+)')
        pattern1 = re.compile(r'([\d]+)-([\d]+)-([\d]+)')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            clothes_id = str(pid)+'-'+ pattern1.search(img_path).group(3)
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(list(pid_container))
        clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            mask_path = osp.join(self.mtrain_dir, osp.basename(img_path)[:-4]+ '.png')
            clothes = str(pid)+'-'+ pattern1.search(img_path).group(3)
            # pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            description,des_inv,des_cloth=self.cap_gen(imgdir2attribute[img_path])
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 512  # pid == 0 means background
            assert 1 <= camid <= 4
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, clothes_id, camid,torch.tensor(imgdir2attribute[img_path]).numpy(),description,des_inv,des_cloth,mask_path))
            # dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        num_clothes = len(clothes_container)
        return dataset,num_pids, num_imgs, num_clothes

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
