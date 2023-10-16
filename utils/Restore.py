import os
import torch
from config import settings


def load(args, model, filename='checkpoint.pth.tar'):
    if args.dataset == "librispeech":
        savedir = os.path.join(settings.update_pth, args.dataset, args.arch, args.seed)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist." % snapshot
    # 判断这个路径是否存在, 存在则不执行任何语句, 不存在则会报出 'snapshot file不存在'

    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    # 用来加载torch.save保存的文件
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('Loaded weights from %s' % snapshot)


def save_model(args, model, is_best=False, filename='checkpoint.pth.tar'):
    state = {'model_state_dict': model.state_dict()}
    if args.dataset == "librispeech":
        savedir = os.path.join(settings.update_pth, args.dataset, args.arch, args.seed)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, 'Sess'+str(args.sess)+filename)
    torch.save(state, savepath)
    print('save weights at %s' %savepath)
    # if is_best:
    #     shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))
