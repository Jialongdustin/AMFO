import csv
import sys, os
sys.path.append(os.getcwd())
from config import settings
import os
import librosa
import numpy as np
import torch
import h5py
import random


class NC_librispeech():
    def __init__(self, args, c_way=5, k_shot=5):
        self.name = 'NC_librispeech'
        self.IndexDir = os.path.join(settings.librispeech_Index_DIR, args.seed)
        self.count = args.sess
        self.args = args
        self.Set_Session(args)

    def Set_Session(self, args):
        self.args = args
        self.sess = args.sess
        self.sess_voice, self.sess_label = self.Read_Index_Sess()
        self.len = len(self.sess_voice)

    def get_data(self):

        return self.sess_voice, self.sess_label

    def Read_Index_Sess(self):
        voice = []
        label = []
        index = []
        true_index = []
        if self.sess == 0:
            train_file = './dataset/librispeech_fscil/mel_features/train1.h5'
        else:
            train_file = './dataset/librispeech_fscil/mel_features/new_train.h5'
        hf = h5py.File(train_file, 'r')
        f = open(self.IndexDir + '/session_' + str(self.sess + 1) + '.csv', 'r')
        reader = csv.reader(f)  # 读取f的一行数据
        if self.sess == 0:
            for i in reader:
                voice.append(hf[b'mel_spectrum'][int(i[0])])
                label.append(hf[b'label'][int(i[0])])
        else:
            for i in reader:
                index.append(int(i[0]))
            for i in range(self.args.ways):
                data_index = random.sample(index[i*500:(i+1)*500], self.args.shots)
                for i in data_index:
                    true_index.append(i)
            for i in true_index:
                voice.append(hf[b'mel_spectrum'][i])
                label.append(hf[b'label'][i])

        return voice, label

    def load_frame(self, index):
        Voice = self.sess_voice[index]
        Label = self.sess_label[index]
        Voice = torch.tensor(Voice)
        Label = torch.tensor(int(Label))

        return Voice, Label

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        Voice, Label = self.load_frame(index)

        return Voice, Label