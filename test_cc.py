"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from evaluate.metrics import evaluate
from test import get_distmat
from evaluate.metrics_for_cc import evaluate_ltcc, evaluate_prcc_all_gallery
from evaluate.re_ranking import re_ranking
from sklearn import datasets
from openTSNE import TSNE
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import MinMaxScaler
from utils.reidtoos import visualize_ranked_results
import os.path as osp
import random

def get_data_for_cc(datasetloader, use_gpu, model,ViT_model):
    with torch.no_grad():
        feats, pids, clothids, camids= [], [], [], []
        for batch_idx, (imgs, pid, clothid, camid, attrlabel,des,des_inv,des_cloth) in enumerate(tqdm(datasetloader)):
            flip_imgs = torch.flip(imgs, [3])
            if use_gpu:
                imgs , flip_imgs= imgs.cuda(), flip_imgs.cuda()
                # imgs=imgs.cuda()
            
            feat, outputs = model(imgs,ViT_model,des,des_inv,des_cloth,pid,clothid)
            feat_flip,_ = model(flip_imgs,ViT_model,des,des_inv,des_cloth,pid,clothid)
            feat += feat_flip
            feat = F.normalize(feat, p=2, dim=1)
            feat = feat.data.cpu()
            feats.append(feat)
            pids.extend(pid)
            clothids.extend(clothid)
            camids.extend(camid)
        feats = torch.cat(feats, 0)
        pids = np.asarray(pids)
        clothids = np.asarray(clothids)
        camids = np.asarray(camids)
    return feats, pids, clothids, camids


def test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model,ViT_model,
                  use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    ViT_model.eval()
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model,ViT_model)
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_sc_loader, use_gpu, model,ViT_model)
    

    # distmat = get_distmat(qf, gf)
    if args.reranking:
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        print("With Reranking: ", end='')
    
    cmc, mAP = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_cc_loader, use_gpu, model,ViT_model)
    # distmat = get_distmat(qf, gf)
    if args.reranking:
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        print("With Reranking: ", end='')
    
    cmc_2, mAP_2 = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP_2), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    print()
    
     # if visrank:
    # upid=np.unique(q_pids)
    # selpid=torch.randperm(len(upid))[:20].detach().numpy()
    # ind=[]
    # for i in range(20):
    #     ind.append(np.where(q_pids == selpid[i])[0][0])
    # query=query_cc_loader.dataset.dataset[ind]
    
    
    
    # upid=np.unique(q_pids)
    # # selpid=torch.randperm(len(upid))[:6].detach().numpy()
    # selpid=[5,10,15,20,25,30,35,40,45,50,55,60]
    # hex1 =["#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF","#FFA500","#800080","#008000", "#000080","#800000","#00FFFF"]#["#FF5733","#33FF57","#3357FF","#F3FF33","#FF33F3","#33FFF3"] #["#c957db", "#dd5f57","#b9db57","#57db30","#5784db","#FE420F"]
    # featq,featg,qlabel,colorlb,gcolorlb=[],[],[],[],[]
    # cl,clclo=0,0
    # for i in selpid:
    #     ind=np.where(q_pids==i)[0]
    #     for j in ind[:10]:
    #         featq.append(qf[j])
    #         qlabel.append(i)
    #         colorlb.append(hex1[cl])
    #     cl+=1
    #     # featq.append(qf[ind[0]])
    #     # colorlb.append(hex1[cl])
    #     # cl+=1
        
    # for i in selpid:
    #     indclo=np.where(g_pids==i)[0]
    #     for j in indclo:
    #         featg.append(gf[j])
    #         gcolorlb.append(hex1[clclo])
    #     clclo+=1
        
    # featq=torch.stack(featq,dim=0)
    # featg=torch.stack(featg,dim=0)
    # featqg=torch.cat((featq,featg),dim=0)
    # # qlabel=torch.tensor(qlabel)
    # N=featq.shape[0]
    
   
    # tsne = TSNE(
    #     perplexity=30,
    #     # init='pca',
    #     n_iter=1000,
    #     metric="euclidean",#cosine
    #     # callbacks=ErrorLogger(),
    #     n_jobs=8,
    #     random_state=42,
    # )
    # # featqg = F.normalize(featqg , p=2, dim=0)
    # # featqg=featqg.detach().numpy()
    # # scaler = MinMaxScaler()
    # # featqg = scaler.fit_transform(featqg)
    # embedding = tsne.fit(featqg)
    # # embedding = TSNE().fit(featqg)
    # # gembedding = TSNE().fit(featg)
    # print(embedding)
    
    # # pyplot.figure(figsize=(10, 6))
    # fig, ax = pyplot.subplots()
    
    # ax.scatter(embedding[:N, 0], embedding[:N, 1], 5,colorlb) # 20： 圆的大小， labels： 颜色
    # ax.scatter(embedding[N:, 0], embedding[N:, 1], 5,gcolorlb,'^')
    # # pyplot.plot(embedding[:N, 0],embedding[:N, 1],colorlb,'o',embedding[N:, 0],embedding[N:, 1],gcolorlb,'d')
    # # pyplot.xlim(-40,40)
    # # pyplot.ylim(-40,40)
    # pyplot.savefig("tsn_003.jpg") # 保存图像
    # pyplot.show()
    
   

    return [cmc[0], cmc_2[0]], [mAP, mAP_2]


def test_for_ltcc(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    ViT_model.eval()
    # ViT_model_m.eval()
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_loader, use_gpu, model,ViT_model)
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model,ViT_model)
    # distmat = get_distmat(qf, gf)
    if args.reranking:
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        print("With Reranking: ", end='')
    else:
        distmat = get_distmat(qf, gf)
        
    for i in range(distmat.shape[0]):
        for j in range(distmat.shape[1]):
            if q_camids[i]==g_camids[j]:
                distmat[i,j] +=0.3
        
    cmc, mAP = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
                             ltcc_cc_setting=False)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    cmc_2, mAP_2 = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
                             ltcc_cc_setting=True)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP_2), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    print()
    
    # save_dir='./visual'
    # dataset_name='LTCC_base'
    # visrank_topk=5
    # visualize_ranked_results(
    #     distmat,
    #     (query_loader.dataset.dataset,gallery_loader.dataset.dataset),#self.datamanager.fetch_test_loaders(dataset_name)
    #     "image",#self.datamanager.data_type
    #     width=128,#self.datamanager.width
    #     height=256,#self.datamanager.height
    #     save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
    #     topk=visrank_topk
    # )
    
    ###############################################################################
    # upid=np.unique(q_pids)
    # # selpid=random.sample(list(upid), 6)#torch.randperm(len(upid))[:6].detach().numpy()
    # selpid=[83,84,105,9,29,145]#6,9,11,37,29,31#37,59,60,79,83,84,143
    # hex1 =["#FF5733","#33FF57","#3357FF","#F3FF33","#FF33F3","#33FFF3"]#["#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF","#FFA500","#800080","#008000", "#000080","#800000","#00FFFF"]# #["#c957db", "#dd5f57","#b9db57","#57db30","#5784db","#FE420F"]
    # gfeatsame,gsamelb,gfeatclo,gclolb=[],[],[],[]
    # cl,clclo=0,0
        
    # for i in selpid:
    #     indclo=np.where(g_pids==i)[0]
    #     ind=np.where(q_pids==i)[0]
    #     for j in indclo:
    #         if q_clothids[ind[0]]==g_clothids[j] and g_camids[j] != q_camids[ind[0]]:
    #             gfeatsame.append(gf[j])
    #             gsamelb.append(hex1[clclo])
    #         if q_clothids[ind[0]]!=g_clothids[j] and g_camids[j] != q_camids[ind[0]]:
    #             gfeatclo.append(gf[j])
    #             gclolb.append(hex1[clclo])
    #     clclo+=1
        
    # gfeatsame=torch.stack(gfeatsame,dim=0)
    # gfeatclo=torch.stack(gfeatclo,dim=0)
    # featall=torch.cat((gfeatsame,gfeatclo),dim=0)
    # N=gfeatsame.shape[0]
    
    # # tsne = TSNE(
    # #     perplexity=30,
    # #     # init='pca',
    # #     n_iter=1000,
    # #     metric="euclidean",#cosine
    # #     # callbacks=ErrorLogger(),
    # #     n_jobs=8,
    # #     random_state=42,
    # # )
    # # embedding = tsne.fit(featall)
    # embedding = TSNE().fit(featall)
    # fig, ax = pyplot.subplots()
    # ax.scatter(embedding[:N, 0], embedding[:N, 1], 5,gsamelb) # 20： 圆的大小， labels： 颜色
    # ax.scatter(embedding[N:, 0], embedding[N:, 1], 5,gclolb,'^')
    # pyplot.savefig("tsn_003.jpg") # 保存图像
    # pyplot.show()

    return [cmc[0], cmc_2[0]], [mAP, mAP_2]


def test_for_cc(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    ViT_model.eval()
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_loader, use_gpu, model,ViT_model)
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model,ViT_model)
    if args.reranking:
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        print("With Reranking: ", end='')
    else:
        distmat = get_distmat(qf, gf)

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    return cmc[0], mAP
