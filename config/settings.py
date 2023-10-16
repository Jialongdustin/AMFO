import os
import os.path as osp
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))  # 获取当前脚本的绝对路径

librispeech_Index_DIR = osp.abspath('./data_list/librispeech/Index_list')
librispeech_DIR = osp.abspath('./dataset/librispeech_fscil/spk_segments')

nsynth_Index_DIR = osp.abspath('./data_list/nsynth/Index_list')
nsynth_DIR = './dataset/nsynth/mel_features'

fsc_Index_DIR = osp.abspath('./data_list/fsc_89/Index_list/')
fsc_DIR = './dataset/fsc_89/mel_features'

# Session Length
librispeech_SessLen = [60, 5, 5, 5, 5, 5, 5, 5, 5]
nsynth_SessLen = [55, 5, 5, 5, 5, 5, 5, 5, 5, 5]
fsc_SessLen = [59, 5, 5, 5, 5, 5, 5]

# mel parameters
sr = 44100
preemphasis = .97
n_ffts = 2048
hop_length = 512
win_length = 2048
fmin = 50
fmax = 14000
n_mels = 128
max_db = 100
min_db = 20
# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
LR = 3.5e-4
SNAPSHOT_DIR = os.path.join('snapshots')
update_pth = os.path.join('update_pth')
