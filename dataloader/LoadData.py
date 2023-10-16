from torch.utils.data import DataLoader
from dataloader.librispeech import NC_librispeech
from dataloader.librispeech_val import librispeech_val


def data_loader(args):
    batch = args.batch_size
    if args.dataset == 'librispeech':
        voice_train = NC_librispeech(args)
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

    return val_loader




