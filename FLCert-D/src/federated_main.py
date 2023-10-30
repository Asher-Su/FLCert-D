#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle

import numpy
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference,test_FLcert_D_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args) # 展示参数细节

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    # train_dataset保存训练数据集，test_dataset保存测试数据集，user_groups保存某用户对应的字典集
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    group_model_list=[] # 存储保存的group_model
    # group_model_list_weight=[[]] * args.num_groups
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            # global_model = CNNMnist(args=args)
            for i in range(args.num_groups):
                mo = CNNMnist(args=args)
                group_model_list.append(mo)
        elif args.dataset == 'fmnist':
            # global_model = CNNFashion_Mnist(args=args)
            for i in range(args.num_groups):
                mo = CNNFashion_Mnist(args=args)
                group_model_list.append(mo)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar(args=args)
            for i in range(args.num_groups):
                mo = CNNCifar(args=args)
                group_model_list.append(mo)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                             dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    # global_model.to(device)
    # 设置模型进行训练模式
    # global_model.train()
    # print(global_model)

    # Set the group model to train and send it to device
    for i in range(len(group_model_list)):
        group_model_list[i].to(device)
        # 设置group model为训练模式
        group_model_list[i].train()

    # copy weights
    # 获取模型的参数
    # global_weights = global_model.state_dict()

    # 获取组模型的参数
    # for i in range(len(group_model_list)):
    #     group_model_list_weight[i] = group_model_list[i].state_dict()

    # Training
    all_idxs=[]
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    mali_users = int(args.num_users * args.mal_radio)
    for i in range(args.num_users):
        all_idxs.append(i)
    temp_mali_idxs_users = set(np.random.choice(range(args.num_users), int(args.num_users * args.mal_radio), replace=False))
    idxs_users = numpy.array(list(set(all_idxs) - temp_mali_idxs_users))
    mali_idxs_users = [[i] for i in temp_mali_idxs_users]
    mali_idxs_users = numpy.array(mali_idxs_users)
    print("Benign clients are",idxs_users)
    print("Malicious clients are",mali_idxs_users)

    # 用tqdm库的函数来完成进度条显示的操作
    for epoch in tqdm(range(args.epochs)):
        # 初始化
        local_weights, local_losses, group_weights = [], [], []
        for i in range(args.num_groups):
            local_weights.append([])
            group_weights.append([])
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # 设置模型进入训练状态
        for i in range(len(group_model_list)):
            # 设置group model为训练模式
            group_model_list[i].train()

        for i in range(args.num_users):
            print(i)
            if i in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[i], logger=logger)
                reflect_group = local_model.reflect_group(args.num_groups, i)
                print("Benign: i-->reflect_group: ",i,"----->",reflect_group)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(group_model_list[reflect_group]), global_round=epoch)
                local_weights[reflect_group].append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            else:
                print(user_groups[i])
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[i], logger=logger)
                reflect_group = local_model.reflect_group(args.num_groups, i)
                print("Malicious: i-->reflect_group: ", i, "----->", reflect_group)
                w, loss = local_model.mali_update_weights(
                    model=copy.deepcopy(group_model_list[reflect_group]), global_round=epoch)
                local_weights[reflect_group].append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

        # 计算所有local_model的平均损失值
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # 更新组模型的权值
        for i in range(args.num_groups):
            group_weights[i] = average_weights(local_weights[i])
            group_model_list[i].load_state_dict(group_weights[i])

        # # update global weights
        # # 计算全局模型的权值
        # for i in range(args.num_groups):
        #     if i in vote_agree_list:
        #         global_weights_list.append(group_weights[i])
        #     else:
        #         continue
        # # print("!!!!!!!",global_weights_list)
        # global_weights = average_weights(global_weights_list)
        #
        # # update global weights
        # # 更新全局模型的权值
        # global_model.load_state_dict(global_weights)
        #
        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users):
        # #for i in range(len(idxs_users)):
        #     # 用每一个local_model的测试数据集来评价模型的准确度
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        test_acc,test_loss = [],[]
        for i in range(args.num_groups):
            temp_test_acc, temp_test_loss = test_inference(args,group_model_list[i] , test_dataset)
            test_acc.append(temp_test_acc)
            test_loss.append(temp_test_loss)

        accuracy = test_FLcert_D_inference(args,group_model_list,test_dataset)
        # print(test_FLcert_D_acc)

        for i in range(len(test_acc)):
            print("\nFor the group model ",i)
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc[i]))
        print("\n!The final accuracy is", accuracy)
    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)


    # print(f' \n The {args.epochs} global rounds of training:')
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
