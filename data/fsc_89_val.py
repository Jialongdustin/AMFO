import os
import numpy as np
from config import settings
import csv
import h5py
import torch
import random


class fsc_val():
    def __init__(self, args, c_way=5, k_shot=5):
        self.name = 'NC_fsc_89'
        self.root = settings.fsc_DIR
        self.IndexDir = os.path.join(settings.fsc_Index_DIR, args.seed)
        self.count = 0
        self.args = args
        self.Set_Session(args)

    def Set_Session(self, args):
        self.sess = args.sess
        self.args = args
        self.voice_list_all = []
        self.label_list_all = []
        for sess in range(self.sess + 1):
            self.sess_voice, self.sess_label = self.Read_Index_Sess(sess)
            self.voice_list_all += self.sess_voice
            self.label_list_all += self.sess_label
        self.len = len(self.voice_list_all)
        return self.voice_list_all, self.label_list_all
    
    def Random_select_sample(self, args):
        self.sess = args.sess
        voice_sample = []
        label_sample = []
        class_index_base = []
        index_base = []
        true_index_base = []
        sample_index_base = []
        final_index = []
        test_file = './dataset/fsc_89/mel_features/test.h5'
        for sess in range(self.sess + 1):
            index_inc = []
            sample_index_inc = []
            class_index_inc = []
            true_index_inc = []
            hf = h5py.File(test_file, 'r')
            f = open(self.IndexDir + '/test_' + str(sess + 1) + '.csv', 'r')
            reader = csv.reader(f)
            if sess == 0:
                for i in range(args.base_num):
                    class_index_base.append(i)
                true_index_base = random.sample(class_index_base, 5)
                for i in reader:
                    index_base.append(int(i[0]))
                for i in true_index_base:
                    sample_index_base = random.sample(index_base[i*100:(i+1)*100], 20)
                    for i in sample_index_base:
                        final_index.append(i)
                for i in final_index:
                    voice_sample.append(hf[b'mel_spectrum'][i])
                    label_sample.append(hf[b'label'][int(i)])
            else:
                for i in reader:
                    index_inc.append(int(i[0]))
                for i in range(args.inc_num):
                    class_index_inc.append(i)
                for i in class_index_inc:
                    sample_index_inc = random.sample(index_inc[i*200:(i+1)*200], 20)
                    for i in sample_index_inc:
                        true_index_inc.append(i)
                for i in true_index_inc:
                    voice_sample.append(hf[b'mel_spectrum'][i])
                    label_sample.append(hf[b'label'][int(i)])
        return voice_sample, label_sample

    def Update_Session(self, sess):
        self.sess_voice, self.sess_label = self.Read_Index_Sess(sess)
        self.voice_list_all = self.sess_voice
        self.label_list_all = self.sess_label
        self.len = len(self.voice_list_all)

    def Read_Index_Sess(self, sess):   
        voice = []
        label = []
        test_file = './dataset/fsc_89/mel_features/test128.h5'
        hf = h5py.File(test_file, 'r')
        f = open(self.IndexDir + '/test_' + str(sess + 1) + '.csv', 'r')
        reader = csv.reader(f)
        for i in reader:
           voice.append(hf[b'mel_spectrum'][int(i[0])])
           label.append(hf[b'label'][int(i[0])])

        return voice, label

    def Random_choose(self):
        Index = np.random.choice(self.sess_voice, 1, replace=False)[0]

        return Index

    def load_frame(self, Index):
        Voice = self.sess_voice[Index]
        Label = self.sess_label[Index]
        Voice = torch.tensor(Voice)
        Label = torch.tensor(int(Label))

        return Voice, Label

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        Voice = self.voice_list_all[index]
        Label = self.label_list_all[index]
        Voice = torch.tensor(Voice)
        Label = torch.tensor(int(Label))

        return Voice, Label
