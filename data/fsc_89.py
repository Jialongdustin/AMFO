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

random.seed()
class NC_fsc():
    def __init__(self, args, c_way=5, k_shot=5):
        self.name = 'NC_fsc_89'
        self.root = settings.fsc_DIR
        self.IndexDir = os.path.join(settings.fsc_Index_DIR, args.seed)
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
        index_base = []
        true_index = []
        true_index_base = []
        if self.sess == 0:
            train_file = './dataset/fsc_89/mel_features/train128.h5'
        else:
            train_file = './dataset/fsc_89/mel_features/new_train128.h5'
        hf = h5py.File(train_file, 'r')
        f = open(self.IndexDir + '/session_' + str(self.sess + 1) + '.csv', 'r')
        reader = csv.reader(f)  # 读取f的一行数据
        if self.sess == 0:
            '''
            for i in reader:
                index_base.append(int(i[0]))
            for i in range(59):
                data_index_base = random.sample(index_base[i*800:(i+1)*800], 500)
                for i in data_index_base:
                    true_index_base.append(i)
            for i in true_index_base:
                voice.append(hf[b'mel_spectrum'][i])
                label.append(hf[b'label'][i])
            '''
            for i in reader:
                voice.append(hf[b'mel_spectrum'][int(i[0])])
                label.append(hf[b'label'][int(i[0])])
        else:
            '''
            if self.sess == 1:
                random.seed(4)
            elif self.sess == 2:
                random.seed(13)
            elif self.sess == 3:
                random.seed(15)
            elif self.sess == 4:
                random.seed(19)
            elif self.sess == 5:
                random.seed(19)
            else:
                random.seed(8)
            '''
            for i in reader:
                index.append(int(i[0]))
            for i in range(5):
                data_index = random.sample(index[i*500:(i+1)*500], 5)
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


def get_melspectrum(fpath):
    y, sr = librosa.load(fpath, sr=settings.sr)
    window_size = settings.win_length
    hop_size = settings.hop_length
    window_func = np.hanning(window_size)  # 2048
    melW = librosa.filters.mel(sr=settings.sr,
                               n_fft=settings.n_ffts,
                               n_mels=settings.n_mels,
                               fmin=settings.fmin,
                               fmax=settings.fmax).T  # (n_ffts-1)/2 x n_mels
    stft_matrix = librosa.core.stft(y=y,
                                    n_fft=window_size,
                                    hop_length=hop_size,
                                    window=window_func,
                                    center=True,
                                    dtype=np.complex64,
                                    pad_mode='reflect').T  # 帧数x(n_ffts-1)/2
    mel_spectrum = np.dot(np.abs(stft_matrix)**2, melW)

    log_melspectrum = librosa.core.power_to_db(mel_spectrum, ref=1.0, amin=1e-10, top_db=None)
    log_melspectrum = log_melspectrum.astype(np.float32)
    mel = log_melspectrum.reshape(1, log_melspectrum.shape[0], log_melspectrum.shape[1])
    mel = torch.Tensor(mel)
    
    return mel
