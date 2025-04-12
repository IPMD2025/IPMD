from __future__ import absolute_import
from collections import defaultdict
import numpy as np
import random
import copy


from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler_cc(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.index_dic_clo = defaultdict(lambda: defaultdict(list))
        for index, (_, pid, cloid,_,_, _, _,_,_) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            self.index_dic_clo[pid][cloid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            idxs_clo=[]
            for cloid,_ in self.index_dic_clo[pid].items():
                idxs_clo.append(copy.deepcopy(self.index_dic_clo[pid][cloid]))
            
            l=len(idxs_clo)
            if l==1:
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            # elif l<4:
            else:
                idx_c=random.sample(range(l), 2)
                if len(idxs_clo[idx_c[0]]) < self.num_instances/2:
                    idxs_clo[idx_c[0]] = np.random.choice(idxs_clo[idx_c[0]], size=self.num_instances/2, replace=True)
                if len(idxs_clo[idx_c[1]]) < self.num_instances/2:
                    idxs_clo[idx_c[1]] = np.random.choice(idxs_clo[idx_c[1]], size=self.num_instances/2, replace=True)
                
                random.shuffle(idxs_clo[idx_c[0]])
                random.shuffle(idxs_clo[idx_c[1]])
                batch_idxs = []
                for idx in range(min(len(idxs_clo[idx_c[0]]),len(idxs_clo[idx_c[1]]))):
                    batch_idxs.append(idxs_clo[idx_c[0]][idx])
                    batch_idxs.append(idxs_clo[idx_c[1]][idx])
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            # else:
            #     if len(idxs) < self.num_instances:
            #         idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            #     random.shuffle(idxs)
            #     batch_idxs = []
            #     for idx in idxs:
            #         batch_idxs.append(idx)
            #         if len(batch_idxs) == self.num_instances:
            #             batch_idxs_dict[pid].append(batch_idxs)
            #             batch_idxs = [] 
            
            # else:
            #     if len(idxs_clo[0]) <= self.num_instances/4:
            #         idxs_clo[0] = np.random.choice(idxs_clo[0], size=self.num_instances/4, replace=True)
            #     if len(idxs_clo[1]) <= self.num_instances/4:
            #         idxs_clo[1] = np.random.choice(idxs_clo[1], size=self.num_instances/4, replace=True)
            #     if len(idxs_clo[2]) <= self.num_instances/4:
            #         idxs_clo[2] = np.random.choice(idxs_clo[2], size=self.num_instances/4, replace=True)
            #     if len(idxs_clo[3]) <= self.num_instances/4:
            #         idxs_clo[3] = np.random.choice(idxs_clo[3], size=self.num_instances/4, replace=True)
                
            #     random.shuffle(idxs_clo[0])
            #     random.shuffle(idxs_clo[1])
            #     random.shuffle(idxs_clo[2])
            #     random.shuffle(idxs_clo[3])
            #     batch_idxs = []
            #     for idx in range(min(len(idxs_clo[0]),len(idxs_clo[1]),len(idxs_clo[2]),len(idxs_clo[3]))):
            #         batch_idxs.append(idxs_clo[0][idx])
            #         batch_idxs.append(idxs_clo[1][idx])
            #         batch_idxs.append(idxs_clo[2][idx])
            #         batch_idxs.append(idxs_clo[3][idx])
            #         if len(batch_idxs) == self.num_instances:
            #             batch_idxs_dict[pid].append(batch_idxs)
            #             batch_idxs = [] 
           
           

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
