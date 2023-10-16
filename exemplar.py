import random
import torch
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as Image
import collections
from collections import Counter
import time
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread


class MyThread(Thread):  # 定义这一类为线程类
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = self.func(self.args)

    def get_result(self):
        return self.result


class Exemplar:
    def __init__(self, args):
        self.mean = {}
        self.cov = {}
        self.val = {}
        self.dataset = args.dataset
        self.k_shot = args.sample_k  # 5
        self.base_num = args.base_num
        self.p = torch.ones(args.label_num).long() * self.k_shot   # 旧类分布采样中的样本特征数
        self.newsample_num = args.newsample_num  # 5
        self.oldsample_num_min = args.oldsample_num_min  # 5
        self.basesample_num_min = args.basesample_num_min  # 5

    def update(self, memory_mean, memory_cov):
        lam = 0.8
        for key in memory_mean.keys():

            if key not in self.mean.keys():
                self.mean[key] = memory_mean[key]
                self.cov[key] = memory_cov[key]
            else:  # 如果是相同类的特征数据,则新的均值,协方差数据由两者共同更新
                self.mean[key] = lam * self.mean[key] + (1 - lam) * memory_mean[key]
                self.cov[key] = lam * self.cov[key] + (1 - lam) * memory_cov[key]
        
    def get_exemplar_train(self):
        # exemplar_feature, exemplar_label = self.multi_process_sampling()
        exemplar_feature, exemplar_label = self.multi_thread_sampling()
        # exemplar_feature, exemplar_label = self.general_sampling()
        return exemplar_feature, exemplar_label

    def sampling(self, key):
        exemplar_feature = []
        exemplar_label = []
        ger_mean = self.mean[key]
        ger_cov = self.cov[key]
        if key >= self.memory_lidx:
            ger_num = self.newsample_num   # 新增加的类
        elif (key < self.memory_lidx) and (key >= self.base_num):
            if self.dataset == 'librispeech':
                # ger_num = min(self.oldsample_num_min, self.p[key].item())
                ger_num = max(self.oldsample_num_min, self.p[key].item())
                # 不是基类,是上一阶段的新类,这一阶段的旧类
            else:
                ger_num = max(self.oldsample_num_min, self.p[key].item())
        else:
            if self.dataset == 'librispeech':
                # ger_num = min(self.basesample_num_min, self.p[key].item())  # 基类
                ger_num = max(self.basesample_num_min, self.p[key].item())
            else:
                ger_num = max(self.basesample_num_min, self.p[key].item())
        
        # ger_feature = self.vae(ger_mean, ger_cov)
        ger_feature = np.random.multivariate_normal(mean=ger_mean, cov=ger_cov, size=ger_num)
        # 生成一个多元正态矩阵(一维均值,协方差矩阵,生成矩阵的大小),size大小为采样的样本数x节点数
        # 每次生成的特征数据都是不同的
        ger_feature = np.float32(ger_feature)
        for i in range(ger_num):
            exemplar_feature.append(ger_feature[i].squeeze())
            # squeeze的作用是对tensor变量进行维度压缩, 去除维数为1的维度
            # unsqueeze的作用是对数据维度进行扩充, 给指定位置加上维数为1的维度.
            if key >= self.memory_lidx:
                exemplar_label.append(-1 * key)   # 将采样出的新增加的类标签值设为负数
            else:
                exemplar_label.append(key)

        self.p[key] = self.k_shot
        return exemplar_feature, exemplar_label
    
    def vae(self, mean, cov):
        feature = []
        for i in range(5):
            epsilon = np.array(torch.randn(mean.shape))
            sample = mean + np.sqrt(cov)*epsilon
            feature.append(sample)
        
        return feature

    def multi_process_sampling(self, processes=5):
        exemplar_feature = []
        exemplar_label = []
        pool = ThreadPool(processes=processes)
        #  pfunc = partial(func, param1)
        out = pool.map(self.sampling, self.mean.keys())
        # map函数的原型是map(function, iterable, …),它的返回结果是一个列表.将function应用于iterable的每一个元素，结果以列表的形式返回
        # 参数function传的是一个函数名,可以是python内置的,也可以是自定义的
        # 参数iterable传的是一个可以迭代的对象,例如列表,元组,字符串这样的
        for out_iter in out:
            exemplar_feature.extend(out_iter[0])
            exemplar_label.extend(out_iter[1])
        pool.close()
        pool.join()

        return exemplar_feature, exemplar_label

    def general_sampling(self):
        exemplar_feature = []
        exemplar_label = []
        for key in self.mean.keys():
            out = self.sampling(key)
            exemplar_feature.extend(out[0])
            exemplar_label.extend(out[1])
        return exemplar_feature, exemplar_label

    def multi_thread_sampling(self):
        exemplar_feature = []
        exemplar_label = []
        threads = []
        # key_len = len(self.mean.keys())
        for key in self.mean.keys():
            t = MyThread(self.sampling, key)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()  # 实现线程同步
        for t in threads:
            out = t.get_result()
            exemplar_feature.extend(out[0])
            exemplar_label.extend(out[1])
        return exemplar_feature, exemplar_label

    def get_len(self):
        return len(self.train)


class BatchData(Dataset):
    def __init__(self, args, voices, labels, IOF='voice'):
        self.voices = voices
        self.labels = labels
        self.IOF = IOF
        self.args = args

    def __getitem__(self, index):
        voice = self.voices[index]
        label = self.labels[index]
        
        return voice, label

    def __len__(self):
        return len(self.voices)


class Distribution:
    def __init__(self):
        self.all_label = torch.zeros(20)
        self.hit_label = torch.zeros(20)
        self.alpha_cov = collections.defaultdict(list)
        self.alpha_sample = collections.defaultdict(list)

    def statistic(self, args, pred, label, label_tmp, output, memory_lidx):
        for k in range(len(pred)):  # len(pred)为总的样本数
            if label[k].item() < args.base_num:
                k_idx = 0
            else:
                k_idx = (label[k].item() - args.base_num) // args.ways + 1  # k_idx表示是第几个session

            # calculate cov's alpha and sample strategy
            if (k_idx == args.sess) and (label_tmp[k].item() >= 0):   # 属于新增加的类, 且不属于sampling中的类
                x = output[k].clone().detach()
                tmp = torch.zeros(args.base_num + 1)
                tmp[:args.base_num] = x[:args.base_num]   # 取新类样本输入第一次节点扩充前网络的预测结果
                tmp[args.base_num] = x[label[k].item()]   # 该类样本在扩充后节点上的概率值
                tmp = torch.softmax(tmp, dim=0)           # 将基类和当前新类的概率值归一化
                self.alpha_cov[label[k].item()].append(tmp.cpu().numpy())# 新类在基类分类器上的预测结果

                tmp2 = torch.zeros(memory_lidx + 1)
                tmp2[:memory_lidx] = x[:memory_lidx]    # 取新类样本输入本次节点扩充前网络的特征结果
                tmp2[memory_lidx] = x[label[k].item()]  # 该样本在扩充节点上的概率值
                tmp2 = torch.softmax(tmp2, dim=0)
                self.alpha_sample[label[k].item()].append(tmp2.cpu().numpy())
                # 决定下一次采样中对old_session的采样样本数量


    def statistic_cov(self, args, memory_lidx, memory_cov, base_cov):
        for i in range(memory_lidx, memory_lidx + args.ways):
            alpha_tmp = np.array(self.alpha_cov[i])
            alpha_tmp = np.mean(alpha_tmp, axis=0)
            memory_cov[i] = memory_cov[i] * alpha_tmp[args.base_num]
            for j in range(args.base_num):
                memory_cov[i] = memory_cov[i] + base_cov[j] * alpha_tmp[j]
            # 利用旧类的协方差和新类的预测结果的均值计算新类的协方差
        return  memory_cov

    def statisitc_sample(self, args, exemplar, k):
        for i in range(exemplar.memory_lidx, exemplar.memory_lidx + args.ways):
            sample_tmp = np.array(self.alpha_sample[i])
            sample_tmp = np.mean(sample_tmp, axis=0)   # 新类输出特征的均值向量
            sample_val, sample_idx = torch.topk(torch.from_numpy(sample_tmp[:exemplar.memory_lidx]), k)
            # torch.topk函数返回最大的k个值,实验中设定的k为5     将numpy.ndarray转化为Tensor
            for j in range(k):
                if sample_val[j] >= max(0, sample_tmp[exemplar.memory_lidx]):
                    # 如果新类在扩充前网络的预测概率大于在扩充后网络的预测概率, 则对这一类别的采样样本数＋1
                    exemplar.p[sample_idx[j]] = exemplar.p[sample_idx[j]] + 1
