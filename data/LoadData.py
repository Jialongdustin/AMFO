from torch.utils.data import DataLoader
from data.librispeech import NC_librispeech
from data.librispeech_val import librispeech_val
from data.nsynth import NC_nsynth
from data.nsynth_val import nsynth_val
from data.fsc_89 import NC_fsc
from data.fsc_89_val import fsc_val

def data_loader(args):
    batch = args.batch_size
    if args.dataset == 'librispeech':
        voice_train = NC_librispeech(args)
        if args.sesses > 0:
            return voice_train
        else:
            train_loader = DataLoader(voice_train, batch_size=batch, shuffle=True, num_workers=8)
    elif args.dataset == 'nsynth':
        voice_train = NC_nsynth(args)
        if args.sesses > 0:
            return voice_train
        else:
            train_loader = DataLoader(voice_train, batch_size=batch, shuffle=True, num_workers=8)
    else:
        voice_train = NC_fsc(args)
        if args.sesses > 0:
            return voice_train
        else:
            train_loader = DataLoader(voice_train, batch_size=batch, shuffle=True, num_workers=8)
    return train_loader


def val_loader(args, val):
    batch = args.batch_size
    if args.dataset == 'librispeech':
        voice_val = librispeech_val(args, val=val)
        val_loader = DataLoader(voice_val, batch_size=batch, shuffle=False, num_workers=8)
    # 从数据集中拿数据, batch_size为每次取多少数据, num_workers决定了有几个进程来处理data loading
    elif args.dataset == 'nsynth':
        voice_val = nsynth_val(args)
        val_loader = DataLoader(voice_val, batch_size=batch, shuffle=False, num_workers=8)
    else:
        voice_val = fsc_val(args)
        val_loader = DataLoader(voice_val, batch_size=batch, shuffle=False, num_workers=8)

    return val_loader




