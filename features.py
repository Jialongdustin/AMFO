import h5py
import json
import os
import torch
import os.path as osp
import librosa
from config import settings
import numpy as np
import pandas as pd


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
dataset = 'librispeech'
voice_train = []
label_train = []
voice_val = []
label_val = []
voice_test = []
label_test = []
if dataset == 'librispeech':
    # Read train,val,test.csv
    librispeech_Index_DIR_train = osp.abspath('./dataset/librispeech_fscil/librispeech_fscil_train.csv')
    librispeech_Index_DIR_val = osp.abspath('./dataset/librispeech_fscil/librispeech_fscil_val.csv')
    librispeech_Index_DIR_test = osp.abspath('./dataset/librispeech_fscil/librispeech_fscil_test.csv')
    librispeech_DIR = osp.abspath('./dataset/librispeech_fscil/spk_segments')
    train_file = pd.read_csv(librispeech_Index_DIR_train, header=None)
    val_file = pd.read_csv(librispeech_Index_DIR_val, header=None)
    test_file = pd.read_csv(librispeech_Index_DIR_test, header=None)
    aduio = train_file[0]
    label = train_file[2]
    # generate h5py path and document to store mel_fratures and labels
    workspace = './dataset/librispeech_fscil/mel_features'
    train_name = 'train.h5'
    new_train_name = 'new_train.h5'
    val_name = 'val.h5'
    test_name = 'test.h5'
    new_train_file = osp.join(workspace, new_train_name)
    val_file = osp.join(workspace, val_name)
    train_file =  osp.join(workspace, train_name)
    test_file = osp.join(workspace, test_name)
    hf1 = h5py.File(train_file, 'w')
    hf2 = h5py.File(test_file, 'w')
    hf3 = h5py.File(val_file, 'w')
    hf4 = h5py.File(new_train_file, 'w')
    # extract mel_features
    for i in range(len(aduio)-30000):
        fpath = os.path.join(librispeech_DIR, aduio[i+30000])
        mel = get_melspectrum(fpath)
        voice_val.append(mel)
        label_val.append(label[i+30000])
    hf4.create_dataset(
        name='mel_spectrum',
        data=[np.float32(voice) for voice in voice_val])
    hf4.create_dataset(
        name='label',
        data=[label for label in label_val])
    print('generate new_train.h5 down')