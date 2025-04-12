"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import absolute_import

import time
import datetime
import numpy as np
import os.path as osp

import torch
from torch import nn

from utils.arguments import get_args, set_log, print_args, set_gpu
from utils.util import set_random_seed
from dataset import dataset_manager, PRCC
from data_process import dataset_loader_cc, dataset_loader
from losses.triplet_loss import TripletLoss
from losses.cross_entropy_loss import CrossEntropyLabelSmooth,CrossEntropyLoss
from losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss,CosFaceLoss
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from utils.util import load_checkpoint, save_checkpoint
from model import fire,base_block
from model.base_block import *
import train_fire, test_cc, test
from clipS import clip
from clipS.model import *
from torch import optim
from scheduler.scheduler_factory import create_scheduler
device = "cuda" if torch.cuda.is_available() else "cpu"
ViT_model, ViT_preprocess = clip.load("ViT-L/14", device=device,download_root='/media/backup/**/pretrained/') 


def main():
    args = get_args()
    set_log(args)
    print_args(args)
    use_gpu = set_gpu(args)
    set_random_seed(args.seed, use_gpu)

    print("Initializing dataset {}".format(args.dataset))
    if args.dataset == 'prcc':
        dataset = PRCC.PRCC(dataset_root=args.dataset_root, dataset_filename=args.dataset_filename)
        train_loader, query_sc_loader, query_cc_loader, gallery_loader = \
            dataset_loader_cc.get_prcc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    elif args.dataset in ['ltcc', 'deepchange', 'last']:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader_cc.get_cc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    else:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader.get_dataset_loader(dataset, args=args, use_gpu=use_gpu)

    num_classes = dataset.num_train_pids
    pid2clothes = torch.from_numpy(dataset.pid2clothes)
    clothes_classifier = NormalizedClassifier(768,dataset.num_train_clothes)
    # model = fire.FIRe(pool_type='maxavg', last_stride=1, pretrain=True, num_classes=num_classes)
    model = TransformerClassifier(num_classes=num_classes)
    # classifier = fire.Classifier(feature_dim=model.feature_dim, num_classes=num_classes)

    # class_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=use_gpu)
    class_criterion = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=False)
    metric_criterion = TripletLoss(margin=args.margin)
    # FFM_criterion = fire.AttrAwareLoss(scale=args.temperature, epsilon=args.epsilon)
    
    criterion_clothes=CosFaceLoss(scale=args.LOSS_CLA_S, margin=0)#CrossEntropyLabelSmooth(num_classes=dataset.num_train_clothes,epsilon=0.1, use_gpu=use_gpu)#
    criterion_adv=ClothesBasedAdversarialLoss(scale=args.LOSS_CLA_S, epsilon=args.LOSS_EPSILON)

    # parameters = list(model.parameters()) + list(classifier.parameters())
    # optimizer = torch.optim.Adam(params=[{'params': parameters, 'initial_lr': args.lr}],
    #                              lr=args.lr, weight_decay=args.weight_decay)
    
    clip_params=[]
    for name, param in ViT_model.named_parameters():
        if any(keyword in name for keyword in args.clip_update_parameters):
            print(name, param.requires_grad)
            clip_params.append(param)
        else:
            param.requires_grad = False

    lr = args.lr
    epoch_num = args.max_epoch
    optimizer_cc = optim.AdamW([{'params':clothes_classifier.parameters()}], lr=lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW([{'params':params} for params in clip_params]+[{'params':model.parameters()}],lr=lr, weight_decay=args.weight_decay)#,{'params':clip_params[1]}
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)

    start_epoch = args.start_epoch  # 0 by default
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=True)

        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if use_gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1  # start from the next epoch

    # scheduler = WarmupMultiStepLR(optimizer, milestones=args.step_milestones, gamma=args.gamma,
    #                               warmup_factor=args.warm_up_factor, last_epoch=start_epoch - 1)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        # classifier = nn.DataParallel(classifier).cuda()

    # only test
    if args.evaluate:
        print("Evaluate only")
        if args.dataset == 'prcc':
            test_cc.test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        elif args.dataset == 'ltcc':
            test_cc.test_for_ltcc(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        elif args.dataset in ['deepchange', 'last']:
            test_cc.test_for_cc(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        else:
            test.test(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        return 0

    # train
    print("==> Start training")
    start_time = time.time()
    train_time = 0
    best_mAP, best_rank1 = -np.inf, -np.inf
    best_epoch_mAP, best_epoch_rank1 = 0, 0

    flag = False
    best_mAP_2, best_rank1_2 = -np.inf, -np.inf
    best_epoch_mAP_2, best_epoch_rank1_2 = 0, 0

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train_fire.train(args, epoch + 1, dataset, train_loader, model,ViT_model,clothes_classifier, 
                         optimizer,optimizer_cc, scheduler, class_criterion, metric_criterion,criterion_clothes,criterion_adv, use_gpu,pid2clothes)#classifier,
        train_time += round(time.time() - start_train_time)

        # evaluate
        if (epoch + 1) > args.start_eval_epoch and args.eval_epoch > 0 and (epoch + 1) % args.eval_epoch == 0 \
                or (epoch + 1) == args.max_epoch:
            print("==> Test")
            if args.dataset == 'prcc':
                rank1, mAP = test_cc.test_for_prcc(args, query_sc_loader, query_cc_loader,
                                                   gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            elif args.dataset == 'ltcc':
                rank1, mAP = test_cc.test_for_ltcc(args, query_loader, gallery_loader, model,ViT_model,
                                                   use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            elif args.dataset in ['deepchange', 'last']:
                rank1, mAP = test_cc.test_for_cc(args, query_loader, gallery_loader, model,ViT_model,
                                                 use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            else:
                rank1, mAP = test.test(args, query_loader, gallery_loader, model,ViT_model,
                                       use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            if isinstance(rank1, list):
                rank1, rank1_2 = rank1
                mAP, mAP_2 = mAP
                flag = True

            is_best_mAP = mAP > best_mAP
            is_best_rank1 = rank1 > best_rank1
            if is_best_mAP:
                best_mAP = mAP
                best_epoch_mAP = epoch + 1
            if is_best_rank1:
                best_rank1 = rank1
                best_epoch_rank1 = epoch + 1

            if flag:
                is_best_mAP_2 = mAP_2 > best_mAP_2
                is_best_rank1_2 = rank1_2 > best_rank1_2
                if is_best_mAP_2:
                    best_mAP_2 = mAP_2
                    best_epoch_mAP_2 = epoch + 1
                if is_best_rank1_2:
                    best_rank1_2 = rank1_2
                    best_epoch_rank1_2 = epoch + 1

            if args.save_checkpoint:
                model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
                # classifier_state_dict = classifier.module.state_dict() if use_gpu else classifier.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                save_checkpoint({
                    'model_state_dict': model_state_dict,
                    # 'classifier_state_dict': classifier_state_dict,
                    'clip_model': ViT_model.state_dict(),
                    'optimizer_state_dict': optimizer_state_dict,
                    'rank1': rank1,
                    'mAP': mAP,
                    'epoch': epoch,
                }, is_best_mAP, is_best_rank1, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) +
                                                        '_mAP_' + str(round(mAP * 100, 2)) + '_rank1_' + str(
                    round(rank1 * 100, 2)) + '.pth'))

    print("==> Best mAP {:.4%}, achieved at epoch {}".format(best_mAP, best_epoch_mAP))
    print("==> Best Rank-1 {:.4%}, achieved at epoch {}".format(best_rank1, best_epoch_rank1))
    if flag:
        print("==> Best mAP_2 {:.4%}, achieved at epoch {}".format(best_mAP_2, best_epoch_mAP_2))
        print("==> Best Rank-1_2 {:.4%}, achieved at epoch {}".format(best_rank1_2, best_epoch_rank1_2))

    # time using info
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    main()