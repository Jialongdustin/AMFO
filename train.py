import os
import json
import argparse
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import torch
import torch.optim as optim
import torch.nn.functional as F 

from dataloader.LoadData import data_loader
from dataloader.LoadData import val_loader
from utils import Log
from utils import Restore
from models import *
from models.binary_net import BinaryNet
from config import settings
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances


def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')   # 创建一个解析对象
    parser.add_argument("--sesses", type=int, default='0', help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--max_epoch", type=int, default='200')
    parser.add_argument("--batch_size", type=int, default='128')  # 一次训练的样本数
    parser.add_argument("--dataset", type=str, default='librispeech')
    parser.add_argument("--arch", type=str, default='DSN', help='quickcnn, resnet')
    parser.add_argument("--lr", type=float, default=0.1)  # 0.1
    parser.add_argument("--r", type=float, default=15)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--seed", type=str, default='Seed_1')  # Seed_3
    parser.add_argument("--gpu", type=str, default='1')  #
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--decay_epoch", type=int, nargs='+', default=[40, 80, 120, 160])
    parser.add_argument('--eval_freq', default=5, type=int, help='evaluate model frequency')
    parser.add_argument('--temperature', type=int, default=16)
    parser.add_argument('--weight_atten', type=int, default=2.5, help='the weight of atten_loss', choices=[2.5, 5, 10])
    parser.add_argument('--weight_margin', type=int, default=2, help='the weight of margin_loss', choices=[1, 2, 2.5, 5])
    parser.add_argument('--weight_binary', type=float, default=40, help='the weight of binary_loss', choices=[0.4, 4])
    parser.add_argument('--shots', type=int, default=5, help='the count of samples in incemental stage per class')
    parser.add_argument('--ways', type=int, default=5, help='the count of classes in incemental stage per session')

    return parser.parse_args()


def test(args, network, val_data):
    TP = 0.0
    TP_in = 0.0
    TP_com = 0.0
    All = 0.0
    network.eval()
    with torch.no_grad():
        for i, data in enumerate(val_data):
            voice, label = data
            voice, label = np.float32(voice), np.float32(label)
            voice, label = torch.from_numpy(voice), torch.from_numpy(label)
            voice, label = voice.cuda(), label.cuda() 
            out, out_n, output_n, output = network(voice, args.sess)
            logit_b = output.clone()
            logit_new = output_n.clone()
            logit_b = torch.softmax(logit_b, dim=-1)
            logit_new = torch.softmax(logit_new, dim=-1)
            aggeragate = torch.cat([logit_b, logit_new], dim=-1)
            scores = network.attention_forward(aggeragate)
            atten_weight = torch.softmax(scores, dim=-1)
            combain = atten_weight[:, 0:1] * output + atten_weight[:, 1:2] * output_n
            test_loss, test_loss_margin, test_loss_dis, test_loss_atten, test_loss_binary = network.get_loss(args, output_n, output, combain, scores, label)
            _, pred = torch.max(output_n, dim=1)
            _, in_pred = torch.max(output, dim=1)
            _, com_pred = torch.max(combain, dim=1)
        
            TP += torch.eq(pred, label).sum().float().item()
            TP_in += torch.eq(in_pred, label).sum().float().item()
            TP_com += torch.eq(com_pred, label).sum().float().item()
            All += torch.eq(label, label).sum().float().item()
        # 判断元素是否相等 sum为tenor求和 item为输出tensor中的值

        acc = float(TP) / All
        in_acc = float(TP_in) / All
        com_acc = float(TP_com) / All
    network.train()
    return acc, in_acc, com_acc, test_loss


def train(args):
    if args.dataset == 'librispeech':
        args.label_num = 100
        args.base_num = 60
        args.weight_margin = 1
        args.shots = 5
    lr = args.lr
    network = eval(args.arch).OneModel(args)  
    print(network)
    network.cuda()
    for sess in range(args.sesses + 1):
        args.sess = sess
        network.binary_classifier = BinaryNet(2*(args.base_num + args.sess*args.ways)).cuda()
        params_list = [{'params': network.parameters()}]
        optimizer = optim.SGD(params_list, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        train_loader = data_loader(args)
        test_data = val_loader(args, False)
        dataset_len = train_loader.dataset.__len__()
        val_acc = 0
        acc_in = 0
        acc_com = 0
        Best_ACC_com = 0
        Best_epoch = 0
        test_loss = 0
        ACC_list_com = []
        loss_list = []
        begin_time = time.time()
        for epoch in range(args.max_epoch):  
            network.train()
            if epoch in args.decay_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            for i, data in enumerate(train_loader):
                voice, label = data
                voice, label = np.float32(voice), np.int32(label)
                voice, label = torch.from_numpy(voice), torch.from_numpy(label)
                voice, label = voice.cuda(), label.cuda()
                out, out_n, output_n, output= network(voice, args.sess)
                logit_b = output.clone()
                logit_new = output_n.clone()
                logit_b = torch.softmax(logit_b, dim=-1)
                logit_new = torch.softmax(logit_new, dim=-1)
                aggeragate = torch.cat([logit_b, logit_new], dim=-1)
                scores = network.attention_forward(aggeragate)
                atten_weight = torch.softmax(scores, dim=-1)
                combain = atten_weight[:, 0:1] * output + atten_weight[:, 1:2] * output_n
                _, pred = torch.max(combain, dim=1)
                loss, loss_margin, loss_dis, loss_atten, loss_binary = network.get_loss(args, output_n, output, 16*combain, 16*scores, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = torch.eq(pred, label).sum().float().item() / torch.eq(label, label).sum().float().item()
                all_step = int((dataset_len / args.batch_size))
                Time = time.time()
                print('epoch: %d' % epoch, 'step: %d/%d' % (i, all_step), 'train_loss: %f' % loss, 'test_loss: %f' % test_loss, 'ACC_val: %f' % val_acc, 'ACC_val_in: %f' % acc_in, 'ACC_val_com: %f' % acc_com,
                      'acc_train: %f' % acc, 'best_acc: %f' %Best_ACC_com, 'best_epoch: %d' %Best_epoch, 'Time: %f' % ((Time - begin_time) / 60))
                p_st_1 = 'Training_loss: all_loss: %f, loss_margin: %f, loss_atten: %f, loss_dis: %f, loss_binary: %f' % (loss, loss_margin, loss_atten, loss_dis, loss_binary)
                print(p_st_1)

            val_acc, acc_in, acc_com, test_loss = test(args, network, test_data)
            ACC_list_com.append(acc_com)
            loss_list.append(loss.data.item())
            if Best_ACC_com <= acc_com:
                Best_ACC_com = acc_com
                Best_epoch = epoch
                Restore.save_model(args, network, filename='.pth.tar')
                print('Update Best_ACC %f' % Best_ACC_com)
                print('epoch: %d' % epoch, 'acc_val: %f' % acc_com)
            Log.log(args, network, ACC_list_com, 'acc', sup='Sess0')
            Log.log(args, network, loss_list, 'loss', sup='Sess0')
            # Restore.save_model(args, network, filename='.pth.tar')
    print('acc:', Best_ACC_com)
    print('End')
    

if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    train(args)
