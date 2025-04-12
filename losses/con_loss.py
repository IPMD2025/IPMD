import torch
from torch import nn
import torch.nn.functional as F
from utils.util import euclidean_dist

def con_loss(features, features_m,labels):
    loss = 0.0
    unique_labels = torch.unique(labels)
    
    for label in unique_labels:
        # 获取同一行人的所有特征
        mask = (labels == label)
        same_features = features[mask]
        same_features_m = features_m[mask]
        same_features=torch.mean(same_features,dim=0)
        same_features_m=torch.mean(same_features_m,dim=0)
        
        loss += F.mse_loss(same_features, same_features_m)
        # if len(same_person_features) < 2:
        #     continue
        
        # # 计算同一行人不同特征之间的欧式距离平方和
        # diff_matrix = torch.cdist(same_person_features, same_person_features)
        # loss += torch.sum(diff_matrix ** 2) / 2  # 除以2避免重复计算
    
    return loss / len(unique_labels)