import argparse
import json
import os
import time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import collections

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import settings
from dataloader.LoadData import data_loader, val_loader
from exemplar import BatchData, Distribution, Exemplar
from models import *
from models.binary_net import BinaryNet
from utils import Log, Restore
from utils.Counting import Counting_train
from torch.utils.data import DataLoader

decay = 0.5


def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')
    parser.add_argument("--sesses", type=int, default='8', help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--start_sess", type=int, default='1')
    parser.add_argument("--max_epoch", type=int, default='100')  # 180
    parser.add_argument("--batch_size", type=int, default='128')
    parser.add_argument("--dataset", type=str, default='librispeech')
    parser.add_argument("--arch", type=str, default='DSN')  #
    parser.add_argument("--lr", type=float, default=0.1)  # 0.005 0.002
    parser.add_argument("--r", type=float, default=0.1)  # 0.01
    parser.add_argument("--gamma", type=float, default=4)  # 0.2
    parser.add_argument("--lamda", type=float, default=0.1)  # 0.01
    parser.add_argument("--seed", type=str, default='Seed_1')  # 0.01 #Seed_1
    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument("--pretrained", type=str, default='False')
    parser.add_argument("--DS", type=str, default='True', help='Distribution Support')
    parser.add_argument("--delay_estimation", type=int, default='5')
    parser.add_argument("--delay_testing", type=int, default='100')
    parser.add_argument("--newsample_num", type=int, default=5)
    parser.add_argument("--oldsample_num_min", type=int, default=5)
    parser.add_argument("--basesample_num_min", type=int, default=5)
    parser.add_argument("--top_k", type=int, default='5')
    parser.add_argument("--sample_k", type=int, default='5')
    parser.add_argument("--optimizer", type=str, default='part')
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument("--decay_epoch", type=int, nargs='+', default=[50, 100])
    parser.add_argument('--weight_atten', type=int, default=10, help='the weight of atten_loss')
    parser.add_argument('--weight_margin', type=int, default=2, help='the weight of margin_loss')
    parser.add_argument('--weight_binary', type=int, default=0.4, help='the weight of binary_loss')
    parser.add_argument('--shots', type=int, default=5, help='the count of samples in incemental stage per class')
    parser.add_argument('--ways', type=int, default=5, help='the count of classes in incemental stage per session')


    return parser.parse_args()


def test(args, network, val_data):
    TP = 0.0
    TP_in = 0.0
    TP_com = 0.0
    All = 0.0
    network.eval()
    val_data.dataset.Update_Session(0)
    with torch.no_grad():
        for i, data in enumerate(val_data):
            voice, label = data
            voice, label = np.float32(voice), np.int32(label)
            voice, label = torch.from_numpy(voice), torch.from_numpy(label)
            voice, label = voice.cuda(), label.cuda()
            out, out_n, output, in_domain_logits = network(voice, sess=args.sess, Mode='test')
    
            logit_b = in_domain_logits.clone()
            logit_new = output.clone()
            logit_b = torch.softmax(logit_b, dim=-1)
            logit_new = torch.softmax(logit_new, dim=-1)
            aggeragate = torch.cat([logit_b, logit_new], dim=-1)
            scores = network.attention_forward(aggeragate)
            atten_weight = torch.softmax(scores, dim=-1)
            combain = atten_weight[:, 0:1] * in_domain_logits + atten_weight[:, 1:2] * output
            _, pred = torch.max(output, dim=1)
            _, pred_in = torch.max(in_domain_logits, dim=1)
            _, pred_com = torch.max(combain, dim=1)
            TP += torch.eq(pred, label).sum().float().item()
            TP_in += torch.eq(pred_in, label).sum().float().item()
            TP_com += torch.eq(pred_com, label).sum().float().item()
            All += torch.eq(label, label).sum().float().item()
        acc = float(TP) / All
        acc_in = float(TP_in) / All
        acc_com = float(TP_com) / All
        network.train()
    return acc, acc_in, acc_com


def test_continue(args, network, val_data):
    acc_list = []
    acc_list_in = []
    acc_list_com = []
    network.eval()
    for sess in range(args.sess + 1):  
        TP = 0.0
        TP_in = 0.0
        TP_com = 0.0
        All = 0.0
        val_data.dataset.Update_Session(sess)
        with torch.no_grad():
            for i, data in enumerate(val_data):
                voice, label = data
                voice, label = np.float32(voice), np.int32(label)
                voice, label = torch.from_numpy(voice), torch.from_numpy(label)
                voice, label = voice.cuda(), label.cuda()
                out, out_n, output, in_domain_logits = network(voice, args.sess, Mode='test')  
            
                if args.margin == 'atten':
                    logit_b = in_domain_logits.clone()
                    logit_new = output.clone()
                    logit_b = torch.softmax(logit_b, dim=-1)
                    logit_new = torch.softmax(logit_new, dim=-1)
                    aggeragate = torch.cat([logit_b, logit_new], dim=-1)
                    scores = network.attention_forward(aggeragate)
                    atten_weight = torch.softmax(scores, dim=-1)
                    combain = atten_weight[:, 0:1] * in_domain_logits + atten_weight[:, 1:2] * output
                    # combain = 0.3*torch.softmax(output, dim=1) + 0.7*torch.softmax(in_domain_logits, dim=1)
                elif args.margin == 'postive':
                     combain = in_domain_logits
                elif args.margin == 'negative':
                    combain = output
                else:  # both
                    combain = in_domain_logits + output
                _, pred = torch.max(output, dim=1)
                _, pred_in = torch.max(in_domain_logits, dim=1)
                _, pred_com = torch.max(combain, dim=1)
                TP += torch.eq(pred, label).sum().float().item()
                TP_in += torch.eq(pred_in, label).sum().float().item()
                TP_com += torch.eq(pred_com, label).sum().float().item()
                All += torch.eq(label, label).sum().float().item()
            acc = float(TP) / All
            acc_list.append(acc)
            acc_in = float(TP_in) / All
            acc_list_in.append(acc_in)
            acc_com = float(TP_com) / All
            acc_list_com.append(acc_com)
    network.train()
    return acc_list, acc_list_in, acc_list_com


def acc_list2string(acc_list):
    acc_str = ''
    for idx, item in enumerate(acc_list):
        acc_str += 'Sess%d: %.4f, ' % (idx, item)

    return acc_str


def Trans_ACC(args, acc_list):
    if args.dataset == 'librispeech':
        SessLen = settings.librispeech_SessLen
    ACC = 0
    ACC_N = 0  # new session
    ACC_N_this = 0
    ACC_B = acc_list[0]
    ACC_O = 0
    num = 0
    old_num = 0
    new_num = 0
    new_num_this = 0
    for idx, acc in enumerate(acc_list):
        ACC += acc * SessLen[idx]
        num += SessLen[idx]
        
        if idx == args.sess:  
            ACC_N_this += acc * SessLen[idx]
            new_num_this += SessLen[idx]
        if idx > 0:
            ACC_N += acc*SessLen[idx]
            new_num += SessLen[idx]
        else :
            ACC_O += acc*SessLen[idx]
            old_num += SessLen[idx]

        
    ACC = ACC / num
    ACC_N = ACC_N / new_num
    ACC_O =  ACC_O / old_num
    ACC_N_this = ACC_N_this / new_num_this
    return ACC, ACC_N, ACC_B


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_feature(data_loader, model): 
    feature_dict = collections.defaultdict(list)
    feature_dict_in = collections.defaultdict(list)
    model.eval()

    for i, (x, y) in enumerate(data_loader):
        x, y = np.float32(x), np.int32(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            outputs, outputs_in = model.get_feature(x)
        tmp = outputs.clone().detach()  
        tmp_in = outputs_in.clone().detach()
        for out, label in zip(tmp, y):
            x = out.cpu().numpy()  
            feature_dict[label.item()].append(x)
        for out, label in zip(tmp_in, y):
            x = out.cpu().numpy()
            feature_dict_in[label.item()].append(x)

    model.train()
    return feature_dict, feature_dict_in
    


def init_feature_space(args, network):
    exemplar = Exemplar(args)    
    args.sess = 0
    voice_train= data_loader(args)
    train_x, train_y = voice_train.get_data()
    
    base_loader = DataLoader(BatchData(args, train_x, train_y), batch_size=256, shuffle=True,
                             num_workers=8)
    base_feature_n, base_feature = extract_feature(base_loader, network)  
    base_mean = {}
    base_cov = {}
    memory_mean = {}
    memory_cov = {}
    for key in base_feature.keys():  
        feature = np.array(base_feature[key])  
        mean = np.mean(feature, axis=0)        

        memory_mean[key] = mean
        memory_cov[key] = np.cov(feature.T)
        base_mean[key] = mean
        base_cov[key] = np.cov(feature.T)
    exemplar.update(memory_mean, memory_cov)
   
    return exemplar, base_mean, base_cov

def update_feature_space(args, network, exemplar, voice_train, init=False):
    train_x, train_y = voice_train.get_data()
    train_loader = DataLoader(BatchData(args, train_x, train_y), batch_size=128,
                              shuffle=True, num_workers=8)
    dataset_len = train_loader.dataset.__len__()

    # new session's distribution init
    sess_loader = DataLoader(BatchData(args, train_x, train_y), batch_size=128, 
                             shuffle=True, num_workers=8)
    sess_feature_n, sess_feature = extract_feature(sess_loader, network) 
    train_x = []    
    train_y = []    
    memory_mean = {}  
    memory_cov = {}   
    exemplar.memory_lidx = args.base_num + args.ways * (args.sess - 1)  # session1为60
    for key in sess_feature.keys():
        feature = np.array(sess_feature[key])
        mean = np.mean(feature, axis=0)
        # cov = np.cov(feature.T)
        memory_mean[key] = mean
        memory_cov[key] = np.cov(feature.T)
    
        for i in range(len(sess_feature[key])):
            train_x.append(sess_feature[key][i])
            train_y.append(key)

    # new session's distribution save
    exemplar.update(memory_mean, memory_cov)  
    if init:
        exec('network.fc_aux' + str(args.sess + 2) + '.weight.data.copy_(network.fc1.weight.data)')
        
        temp = np.zeros((args.ways, 512))
        for key in memory_mean.keys():
            temp[key - args.base_num - args.sess * args.ways] = memory_mean[key]
           
        fea = torch.tensor(temp).cuda().to(torch.float32)
        fea = network._l2norm(network.fc1(fea), dim=1)
        exec('network.fc' + str(args.sess + 2) + '.weight.data.copy_(fea.data)')
        exec('network.fc_n' + str(args.sess + 2) + '.weight.data.copy_(fea.data)')
       

    return exemplar, train_x, train_y, memory_mean, memory_cov, dataset_len


def train(args):
    result_str = ''
    last_acc_str = ''
    for i in range(1):
        ACC_list = []
        ACC_list_new = []
        ACC_list_new_in = []
        ACC_list_new_com = []
        ACC_list_base = []  
        ACC_list_base_in = []  
        ACC_list_base_com = []
        lr = args.lr
        network = eval(args.arch).OneModel(args)  # fc:fc1  fw:sess-1 fc
        network.cuda()
        network_Old = eval(args.arch).OneModel(args)  # OLD NETWORK
        network_Old.cuda()
        best_model = eval(args.arch).OneModel(args)
        best_model.cuda()
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, dampening=0.5, weight_decay=0)

        print(network)

        log_dir = os.path.join('./log', args.dataset, args.arch, args.seed)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        args.sess = 0
        val_data = val_loader(args, False)   
        if args.start_sess > 0:
            best_model.binary_classifier = BinaryNet(2*args.base_num).cuda()
            network.binary_classifier = BinaryNet(2*args.base_num).cuda()
            network_Old.binary_classifier = BinaryNet(2*args.base_num).cuda()
            Restore.load(args, network, filename='Sess%d' % (args.start_sess - 1) + '.pth.tar')
            exemplar, base_mean, base_cov = init_feature_space(args, network)
            
            args.sess = args.start_sess - 1
            ACC, ACC_in, ACC_com = test(args, network, val_data) 
            ACC_list.append(ACC_com)
            ACC_list_base_com.append(ACC_com)
            print('Sess: %d' % args.sess, 'acc_val: %f' % ACC,'acc_val_in: %f' % ACC_in, 'acc_val_com: %f' % ACC_com)

        # Initialize feature space
        begin_time = time.time()
        best_model.load_state_dict(network.state_dict())   
        for sess in range(args.start_sess, args.sesses):
            args.sess = sess  
            network.load_state_dict(best_model.state_dict())
            network_Old.load_state_dict(network.state_dict())
            network_Old = freeze_model(network_Old)
            network_Old.eval()
            network.binary_classifier = BinaryNet(2*(args.base_num + args.sess*args.ways)).cuda()
            best_model.binary_classifier = BinaryNet(2*(args.base_num + args.sess*args.ways)).cuda()
            network_Old.binary_classifier = BinaryNet(2*(args.base_num + args.sess*args.ways)).cuda()
            if args.optimizer == 'part':
                
                param_list1 = eval('network.fc' + str(args.sess + 2) + '.parameters()')
                param_list2 = eval('network.fc_aux' + str(args.sess + 2) + '.parameters()')
                param_list3 = eval('network.fc_n' + str(args.sess + 2) + '.parameters()')
                optimizer = optim.SGD([{"params": param_list1}, {"params": param_list2}, {"params": param_list3}, {"params": network.binary_classifier.parameters()}], 
                                        lr=lr, momentum=0.9, dampening=0.5, weight_decay=0)
                
            # Update feature space
            voice_train = data_loader(args) 
            exemplar, train_x, train_y, memory_mean, memory_cov, dataset_len = update_feature_space(args, network, exemplar,
                                                                                                voice_train, True)
            
            Best_ACC = 0
            Best_epoch = 0
            Best_ACC_new = 0
            Best_ACC_new_in = 0
            Best_ACC_new_com = 0
            Best_ACC_base = 0
            Best_ACC_base_in = 0
            Best_ACC_base_com = 0
            for epoch in range(args.max_epoch):
                if epoch % args.delay_estimation == 0: 
                    exemplar, train_x, train_y, memory_mean, memory_cov, dataset_len = update_feature_space(args, network,
                                                                                                        exemplar,
                                                                                                        voice_train)
                if args.DS == 'True':
                    # memory sample
                    if epoch % args.delay_estimation == 0:  
                        train_xs, train_ys = exemplar.get_exemplar_train()
                        train_xs.extend(train_x)
                        train_ys.extend(train_y)
                        print('sample_data_len:', len(train_xs))
                        print('sample_label_len:', len(train_ys))  
                else:
                    train_xs = train_x
                    train_ys = train_y

                train_loader = DataLoader(BatchData(args, train_xs, train_ys, IOF='feature'),
                                      batch_size=args.batch_size, shuffle=True, num_workers=8)
                # statistic
                if args.DS == 'True':
                    distribution = Distribution()
                counting_train = Counting_train(args)

                for i, data in enumerate(train_loader):
                    voice_tmp, label_tmp = data
                    voice_tmp = voice_tmp.to(torch.float32)
                    label_tmp = np.array(np.int32(label_tmp))
                    label_tmp = torch.from_numpy(label_tmp).long()
                    voice_tmp, label_tmp = voice_tmp.cuda(), label_tmp.cuda()

                    voice_old = torch.zeros(0, voice_tmp.shape[1]).to(torch.float32).cuda()
                    label_old = torch.zeros(0).long().cuda()
                    voice_new = torch.zeros(0, voice_tmp.shape[1]).to(torch.float32).cuda()
                    label_new = torch.zeros(0).long().cuda()

                    for i in range(len(label_tmp)):
                        if (label_tmp[i].item() < exemplar.memory_lidx) and (label_tmp[i].item() >= 0):
                            voice_old = torch.cat([voice_old, voice_tmp[i].unsqueeze(dim=0)], dim=0)
                            label_old = torch.cat([label_old, label_tmp[i].unsqueeze(dim=0)], dim=0)
                        else:
                            voice_new = torch.cat([voice_new, voice_tmp[i].unsqueeze(dim=0)], dim=0)
                            label_new = torch.cat([label_new, label_tmp[i].unsqueeze(dim=0)], dim=0)
                    label_tmp = torch.cat([label_old, label_new], dim=0)
                    label = torch.abs(label_tmp)  
                    voice = torch.cat([voice_old, voice_new], dim=0)

                    Compression = True
                    if voice_new.shape[0] != 0:
                        out_new, out_new_n, output_newvoice, in_domain_logits_new = network(voice_new, args.sess, epoch, IOF='feature')
                        out = out_new
                        out_n = out_new_n
                        output = output_newvoice
                        in_domain_logits = in_domain_logits_new
                   
                    if voice_old.shape[0] != 0:
                        out_old, out_old_n, output_oldvoice, in_domain_logits_old = network(voice_old, args.sess, epoch, Mode='test', IOF='feature')
                
                        if voice_new.shape[0] == 0:
                            out = out_old
                            out_n = out_old_n
                            output = output_oldvoice
                            in_domain_logits = in_domain_logits_old
                            Compression = False
                        else:
                            out = torch.cat([out_old, out_new], dim=0)
                            out_n = torch.cat([out_old_n, out_new_n], dim=0)
                            output = torch.cat([output_oldvoice, output_newvoice], dim=0)
                            in_domain_logits = torch.cat([in_domain_logits_old, in_domain_logits_new], dim=0)
                       
                    with torch.no_grad():
                        _, _, outputold, outputold_in = network_Old(voice, args.sess - 1, epoch, Mode='test', IOF='feature')
                    logit_b = in_domain_logits.clone()
                    logit_new = output.clone()
                    logit_b = torch.softmax(logit_b, dim=-1)
                    logit_new = torch.softmax(logit_new, dim=-1)
                    aggeragate = torch.cat([logit_b, logit_new], dim=-1)
                    scores = network.attention_forward(aggeragate)
                    atten_weight = torch.softmax(scores, dim=-1)
                    combain = atten_weight[:, 0:1] * in_domain_logits + atten_weight[:, 1:2] * output
                
                    _, pred = torch.max(output, dim=1)  
                    _, pred_in = torch.max(in_domain_logits, dim=1)
                    _, pred_com = torch.max(combain, dim=1)
                    loss, loss_margin, loss_dis, loss_atten, loss_binary = network.get_loss(args, output, in_domain_logits, 16*combain, 16*scores, label, 16.0 * outputold.detach(), 16.0 * outputold_in.detach(), compression=Compression)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    network.finish_train()
                # statistic
                    if args.DS == 'True':
                        distribution.statistic(args, pred_in, label, label_tmp, in_domain_logits, exemplar.memory_lidx)
                
                    counting_train.count(pred_com, label)  
            # Updating distribution statistic
                if args.DS == 'True':
                    # statistic cov
                    memory_cov = distribution.statistic_cov(args, exemplar.memory_lidx, memory_cov, base_cov)
                    exemplar.update(memory_mean, memory_cov)
                    # statisitc sample
                    # distribution.statisitc_sample(args, exemplar, args.top_k)

                # Train accuracy
                Train_ACC_Sess = counting_train.get_train_acc()
                Train_ACC_Sess_str = acc_list2string(Train_ACC_Sess)

                # information
                Time = time.time()
                p_st_1 = 'Training--' + ' Sess: %d' % args.sess + ' epoch: %d' % epoch + '                  ' + '%s' % Train_ACC_Sess_str + ' Time cost: %.2fm\n ' % (
                        (Time - begin_time) / 60)
                p_st_1 += 'Training_loss: all_loss: %f, loss_margin: %f, loss_atten: %f, loss_dis: %f, loss_binary: %f' % (loss, loss_margin, loss_atten, loss_dis, loss_binary)
                print(p_st_1)

                # Test accuracy
                if epoch >= 0:
                    ACC_Sess, ACC_Sess_in, ACC_Sess_com = test_continue(args, network, val_data)
                else:
                    ACC_Sess, ACC_Sess_in, ACC_Sess_com = [0] * (args.sess + 1)
                ACC_Sess_str = acc_list2string(ACC_Sess)
                ACC_Sess_str_in = acc_list2string(ACC_Sess_in)
                ACC_Sess_str_com = acc_list2string(ACC_Sess_com)
                ACC, ACC_N, ACC_B = Trans_ACC(args, ACC_Sess)
                ACC_in, ACC_N_in, ACC_B_in = Trans_ACC(args, ACC_Sess_in)
                ACC_com, ACC_N_com, ACC_B_com = Trans_ACC(args, ACC_Sess_com)

                # information
                Time = time.time()
                p_st_2 = 'Testing--' + ' Sess: %d' % args.sess + ' epoch: %d' % epoch + ' acc_val: %f' % ACC+ ' acc_val_in: %f' % ACC_in+ ' acc_val_com: %f' % ACC_com + ' acc_session:%s' % ACC_Sess_str + ' acc_session_in:%s' % ACC_Sess_str_in + ' acc_session_com:%s' % ACC_Sess_str_com + ' Time cost: %.2fm ' % (
                        (Time - begin_time) / 60) + '\n'
                print(p_st_2)
                print('args.shots:%d, m_p:%.2f, m_n:%.2f' %(args.shots, network.m_p, network.m_n))

                if (Best_ACC <= ACC_com):
                    Best_ACC = ACC_com
                    Best_epoch = epoch
                    Best_ACC_base = ACC_B
                    Best_ACC_base_in = ACC_B_in
                    Best_ACC_base_com = ACC_B_com
                    Best_ACC_new = ACC_N
                    Best_ACC_new_in = ACC_N_in
                    Best_ACC_new_com = ACC_N_com
                    best_mean = memory_mean
                    Best_st = p_st_2
                    best_cov = memory_cov
                    Restore.save_model(args, network, filename='.pth.tar')
                    best_model.load_state_dict(network.state_dict())
                # 更新网络的权重参数,作为下一session的权重文件
                print('best_epoch: %d, best_acc: %f, best_acc_base: %f, best_acc_new: %f' % (Best_epoch, Best_ACC, Best_ACC_base_com, Best_ACC_new_com))
            network.finish_train()
            # Restore.save_model(args, network, filename='.pth.tar')
            
            with open(log_dir + '/log' + args.gpu + '.txt', 'a') as file_obj:
                file_obj.write(Best_st)
            

            ACC_list.append(Best_ACC)
            ACC_list_new.append(Best_ACC_new)
            ACC_list_new_in.append(Best_ACC_new_in)
            ACC_list_new_com.append(Best_ACC_new_com)
            ACC_list_base.append(Best_ACC_base)
            ACC_list_base_in.append(Best_ACC_base_in)
            ACC_list_base_com.append(Best_ACC_base_com)
            Best_ACC_Sess_str = acc_list2string(ACC_list)
            Best_ACC_new_str = acc_list2string(ACC_list_new)
            Best_ACC_new_str_in = acc_list2string(ACC_list_new_in)
            Best_ACC_new_str_com = acc_list2string(ACC_list_new_com)
            Best_ACC_base_str = acc_list2string(ACC_list_base)
            Best_ACC_base_str_in = acc_list2string(ACC_list_base_in)
            Best_ACC_base_str_com = acc_list2string(ACC_list_base_com)
            print('best_all acc:%s' % Best_ACC_Sess_str)
            print('best_new acc:%s' % Best_ACC_new_str)
            print('best_new acc_in:%s' % Best_ACC_new_str_in)
            print('best_new acc_com:%s' % Best_ACC_new_str_com)
            print('best_base acc:%s' % Best_ACC_base_str)
            print('best_base acc_in:%s' % Best_ACC_base_str_in)
            print('best_base acc_com:%s' % Best_ACC_base_str_com)
            exemplar.update(best_mean, best_cov)
        acc_all = sum(ACC_list)
        acc_new= sum(ACC_list_new_com)
        acc_base = sum(ACC_list_base_com)
        aa_all = acc_all / args.sesses
        aa_new = acc_new / (args.sesses - 1)
        aa_base = acc_base / args.sesses
        PD_all = ACC_list[0] - ACC_list[-1]
        PD_new = ACC_list_new_com[0] - ACC_list_new_com[-1]
        PD_base = ACC_list_base_com[0] - ACC_list_base_com[-1]
        print('AA_base:', aa_base, 'PD_base:', PD_base)
        print('AA_new:', aa_new, 'PD_new:', PD_new)
        print('AA_all:', aa_all, 'PD_all:', PD_all)
        print('End')
        last_acc_str += 'args.shots:%d  last_acc:%.5f\n' %(args.shots, ACC_list[-1])
        result_str = result_str + 'args.shots: %d m_p:%.2f m_n:%.2f AA_base:%.5f AA_new:%.5f AA_all:%.5f PD_base:%.5f PD_new:%.5f PD_all:%.5f\n' %(args.shots, network.m_p, network.m_n, aa_base, aa_new, aa_all, PD_base, PD_new, PD_all)
    print(result_str)
    print(last_acc_str)



if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if args.dataset == 'librispeech':
        args.label_num = 100
        args.base_num = 60
        args.ways = 5
        args.shots = 5
        args.sesses = 9
    elif args.dataset == 'nsynth':
        args.label_num = 100
        args.base_num = 55
        args.ways = 5
        args.shots = 5
        args.sesses = 10
    else:
        args.label_num = 89
        args.base_num = 59
        args.ways = 5
        args.shots = 5
        args.sesses = 7
    train(args)
