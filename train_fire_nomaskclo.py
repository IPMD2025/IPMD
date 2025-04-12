"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

import collections
import time
from tqdm import tqdm, trange
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import faiss

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from losses.con_loss import con_loss
from losses.mmdloss import mmd_loss

from data_process import samplers, transform
from data_process.dataset_loader_cc import ImageClothDataset_cc
from data_process.dataset_loader import ImageDataset
from model import fire
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from utils.util import AverageMeter
from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu
from utils.arguments import get_args
args = get_args()

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def init_fg_cluster(args, model, dataset):
    transform_train, transform_test = transform.get_transform(args)

    if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last']:
        train_loader_normal = DataLoader(
            ImageClothDataset_cc(dataset.train, transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )
    else:
        train_loader_normal = DataLoader(
            ImageDataset(dataset.train, transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )

    model.eval()
    dataid_dict = collections.defaultdict(list)
    feat_dict = collections.defaultdict(list)
    with torch.no_grad():
        dataid = 0
        for data in tqdm(train_loader_normal):
            if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last']:
                img, pid, _, camid = data
            else:
                img, pid, camid = data

            feat = model(img.cuda())
            for i in range(img.shape[0]):
                dataid_dict[int(pid[i])].append(dataid)
                dataid += 1
                feat_dict[int(pid[i])].append(feat[i].unsqueeze(0))
    model.train()

    with torch.no_grad():
        num_pids = len(feat_dict.keys())
        num_fg_class_list = []
        dataid2fgid = collections.defaultdict(int)
        fg_center = []
        pseudo_train_dataset = []
        for pid in trange(num_pids):
            fg_feats = F.normalize(torch.concat(feat_dict[pid], dim=0), p=2, dim=1)
            dist_mat = compute_jaccard_distance(fg_feats, k1=args.k1, k2=args.k2, print_flag=False)
            cluster = DBSCAN(eps=args.eps, min_samples=1, metric='precomputed', n_jobs=-1)
            tmp_pseudo_fgids = cluster.fit_predict(dist_mat)

            # assign labels to outliers
            num_fgids = len(set(tmp_pseudo_fgids)) - (1 if -1 in tmp_pseudo_fgids else 0)
            pseudo_fgids = []
            for id in tmp_pseudo_fgids:
                if id != -1:
                    pseudo_fgids.append(id)
                else:  # outlier
                    pseudo_fgids.append(num_fgids)
                    num_fgids += 1

            def generate_cluster_features(labels, feats):
                feat_centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    feat_centers[labels[i]].append(feats[i])
                feat_centers = [torch.stack(feat_centers[fgid], dim=0).mean(0).detach()
                                for fgid in sorted(feat_centers.keys())]  # n_fg [d]
                return torch.stack(feat_centers, dim=0)

            fg_centers = generate_cluster_features(pseudo_fgids, fg_feats)  # [n_fg, d]

            for i in range(len(dataid_dict[pid])):
                dataid2fgid[dataid_dict[pid][i]] = sum(num_fg_class_list) + pseudo_fgids[i]
            num_fg_class_list.append(num_fgids)
            fg_center.append(fg_centers)
            del fg_feats

        # get new train_loader with fine-grained pseudo label
        for dataid, data in enumerate(dataset.train):
            if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last']:
                img, pid, _, camid = data
            else:
                img, pid, camid = data

            pseudo_train_dataset.append((img, pid, dataid2fgid[dataid], camid))

        fg_center = torch.concat(fg_center, dim=0).detach()
        num_fg_classes = sum(num_fg_class_list)
        pid2fgids = np.zeros((num_pids, num_fg_classes))
        fg_cnt = 0
        for i in range(num_pids):
            pid2fgids[i, fg_cnt: fg_cnt + num_fg_class_list[i]] = 1
            fg_cnt += num_fg_class_list[i]
        pid2fgids = torch.from_numpy(pid2fgids)

        return fg_center, pseudo_train_dataset, pid2fgids, num_fg_classes


def train(args, epoch, dataset, train_loader, model, ViT_model,ViT_model_m,
          optimizer, scheduler, class_criterion, metric_criterion,triplet_hard_criterion, lamd,use_gpu):#classifier,
    # fg_start_epoch = args.fg_start_epoch
    # FAR_times = model.module.K_times
    # FAR_weight = args.FAR_weight
    model.train()
    ViT_model.train()
    ViT_model_m.train()
    losses = AverageMeter()
    class_losses = AverageMeter()
    class_losses_m = AverageMeter()
    conlosses = AverageMeter()
    
    htri_losses_0 = AverageMeter()
    htri_losses_1 = AverageMeter()
    htri_losses_2 = AverageMeter()
   
    htri_losses_3 = AverageMeter()

   

    for batch_idx, (imgs, pids, cloth_ids, camid, attrlabel,des,des_inv,des_cloth,mask) in enumerate(tqdm(train_loader)):#
       
        if use_gpu:
            imgs, pids, cloth_ids, attrlabel = imgs.cuda(), pids.cuda(), cloth_ids.cuda(), attrlabel.float().cuda()
        
        features, outputs = model(imgs,ViT_model,des,des_inv,des_cloth,pids,cloth_ids,ViT_model_m,mask)

      
        loss = 0
        # y = classifier(feat)
        class_loss = class_criterion(outputs[0], pids)
        loss += class_loss
        
        
        if args.ablation=='no':
            htri_loss_0 = metric_criterion(features[0], pids)
            htri_loss_1 = metric_criterion(features[1], pids)
            htri_loss_2 = metric_criterion(features[2], cloth_ids)
            htri_loss_3 = triplet_hard_criterion(features[3], pids)
            # if isflag:
            #     loss += htri_loss_0 + 0.1*htri_loss_1 - lamd*htri_loss_2 + htri_loss_3#
            # else:
            loss += htri_loss_3 + htri_loss_0 + htri_loss_1 - lamd*htri_loss_2 #
        elif args.ablation=='allandinv':
            htri_loss_0 = metric_criterion(features[0], pids)
            htri_loss_1 = metric_criterion(features[1], pids)
            htri_loss_3 = triplet_hard_criterion(features[2], pids)
            loss += htri_loss_0 + htri_loss_1 + htri_loss_3#
            htri_loss_2=torch.tensor(100.).cuda()
        elif args.ablation== 'featandall':
            htri_loss_0 = metric_criterion(features[0], pids)
            htri_loss_3 = triplet_hard_criterion(features[1], pids)
            loss +=  htri_loss_0 +htri_loss_3#
            htri_loss_2=torch.tensor(100.).cuda()
            htri_loss_1=torch.tensor(100.).cuda()
        elif args.ablation=='featandclo':
            htri_loss_2 = metric_criterion(features[0], cloth_ids)
            htri_loss_1 = triplet_hard_criterion(features[1], pids)
            htri_loss_3 = triplet_hard_criterion(features[2], pids)
            class_loss_m = class_criterion(outputs[1], pids)
            # conloss = metric_criterion(features[3], pids)
            conloss = con_loss(features[3], features[4], pids)
            # conloss = mmd_loss(features[3], features[4], sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True)
            # Q = features[3].clone().detach()
            # P = features[4].clone()
            # Q = torch.nn.functional.softmax(Q, dim=-1)
            # P = torch.nn.functional.softmax(P, dim=-1)
            # kl_loss = nn.functional.kl_div(torch.log(Q), P, reduction='sum') + nn.functional.kl_div(torch.log(P), Q, reduction='sum')
            loss += htri_loss_3 - lamd*htri_loss_2 + htri_loss_1 + 0.7*conloss+ class_loss_m 
            htri_loss_0=torch.tensor(100.).cuda()
            # htri_loss_1=torch.tensor(100.).cuda()
        elif args.ablation== 'abla-ita':
            htri_loss_0 = metric_criterion(features[0], pids)
            htri_loss_1 = triplet_hard_criterion(features[1], pids)
            htri_loss_2 = metric_criterion(features[2], cloth_ids)
            htri_loss_3 = triplet_hard_criterion(features[3], pids)
            class_loss_m = class_criterion(outputs[1], pids)
            conloss = con_loss(features[1], features[3], pids)
            loss += htri_loss_0 + htri_loss_1 - lamd*htri_loss_2 + htri_loss_3 + class_loss_m + conloss
            
        else :
            htri_loss_3 = triplet_hard_criterion(features[0], pids)
            loss += htri_loss_3
            htri_loss_2=torch.tensor(100.).cuda()
            htri_loss_1=torch.tensor(100.).cuda()
            htri_loss_0=torch.tensor(100.).cuda()
        
        


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        class_losses.update(class_loss.item(), pids.size(0))
        class_losses_m.update(class_loss_m.item(), pids.size(0))
        conlosses.update(conloss.item(), pids.size(0))
        htri_losses_0.update(htri_loss_0.item(), pids.size(0))
        htri_losses_1.update(htri_loss_1.item(), pids.size(0))
        htri_losses_2.update(htri_loss_2.item(), pids.size(0))
        htri_losses_3.update(htri_loss_3.item(), pids.size(0))
        losses.update(loss.item(), pids.size(0))
        # if epoch > fg_start_epoch:
        #     triplet_losses.update(triplet_loss.item(), pid.size(0))
            # FFM_losses.update(FFM_loss.item(), pid.size(0))
            # FAR_losses.update(FAR_loss.item(), pid.size(0))
    # if args.print_train_info_epoch_freq != -1 and epoch % args.print_train_info_epoch_freq == 0:
    #     print('Epoch{0} Cls:{cls_loss.avg:.4f} '
    #           'Tri_1:{htri_losses_1.avg:.4f} Tri_2:{htri_losses_2.avg:.4f} Tri_3:{htri_losses_3.avg:.4f} losses:{losses.avg:.4f} lamd:{lamd:.4f}'.format(
    #         epoch, cls_loss=class_losses, 
    #         htri_losses_1=htri_losses_1, htri_losses_2=htri_losses_2,htri_losses_3=htri_losses_3,losses=losses,lamd=lamd))#Tri_0:{htri_losses_0.avg:.4f} htri_losses_0=htri_losses_0,

    if args.print_train_info_epoch_freq != -1 and epoch % args.print_train_info_epoch_freq == 0:
        print('Epoch{0} Cls:{cls_loss.avg:.4f} Cls_m:{cls_loss_m.avg:.4f} Con:{conlosses.avg:.4f} Tri_0:{htri_losses_0.avg:.4f} '
              'Tri_1:{htri_losses_1.avg:.4f} Tri_2:{htri_losses_2.avg:.4f} Tri_3:{htri_losses_3.avg:.4f} losses:{losses.avg:.4f} lamd:{lamd:.4f}'.format(
            epoch, cls_loss=class_losses,cls_loss_m=class_losses_m , conlosses=conlosses, htri_losses_0=htri_losses_0,
            htri_losses_1=htri_losses_1, htri_losses_2=htri_losses_2,htri_losses_3=htri_losses_3,losses=losses,lamd=lamd))#n_fg:{num_fg_classes}, num_fg_classes=num_fg_classes


    scheduler.step(epoch)
    return htri_losses_3.avg
