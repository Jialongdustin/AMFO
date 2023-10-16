import os
import os.path as osp
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))  

librispeech_Index_DIR = osp.abspath('./data_list/librispeech/Index_list') # get the data_index
librispeech_DIR = osp.abspath('./dataset/librispeech_fscil/spk_segments')

# Session Length
librispeech_SessLen = [60, 5, 5, 5, 5, 5, 5, 5, 5]

