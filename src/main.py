'''
main.py

* Parsing arguments
* Loading/saving checkpoints
* main function
'''

import argparse
import numpy as np
import torch
import os
import random
from src.loader import loader
from src.umodel import StegoUNet
from src.train import train, validate
import torch.nn as nn


# torch.backends.cudnn.benchmark=True
def set_reproductibility(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


### PARSING ###

def parse_keyword(keyword):
    if isinstance(keyword, bool): return keyword
    if keyword.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif keyword.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong keyword.')


parser = argparse.ArgumentParser()
parser.add_argument('--beta',
                    type=float,
                    default=1,
                    metavar='DOUBLE',
                    help='Beta hyperparameter'
                    )
parser.add_argument('--lam',
                    type=float,
                    default=1,
                    metavar='DOUBLE',
                    help='Lambda hyperparameter'
                    )
parser.add_argument('--alpha',
                    type=float,
                    default=1,
                    metavar='DOUBLE',
                    help='Alpha hyperparameter'
                    )
parser.add_argument('--gamma',
                    type=float,
                    default=1,
                    metavar='DOUBLE',
                    help='Gamma hyperparameter'
                    )
parser.add_argument('--dtw',
                    type=parse_keyword,
                    default=False,
                    metavar='BOOL',
                    help='Use DTW (instead of L1) loss'
                    )
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    metavar='DOUBLE',
                    help='Learning rate hyperparameter'
                    )
parser.add_argument('--val_itvl',
                    type=int,
                    default=500,
                    metavar='INT',
                    help='After how many training steps to do validation'
                    )
parser.add_argument('--val_size',
                    type=int,
                    default=50,
                    metavar='INT',
                    help='Steps of every validation round'
                    )
parser.add_argument('--num_epochs',
                    type=int,
                    default=8,
                    metavar='INT',
                    help='Number of training epochs'
                    )
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    metavar='INT',
                    help='Size of the data batch'
                    )
parser.add_argument('--experiment',
                    type=int,
                    default=1,
                    metavar='INT',
                    help='Number of experiment'
                    )
parser.add_argument('--summary',
                    type=str,
                    default="Test_Test",
                    metavar='STRING',
                    help='Summary to be shown in wandb'
                    )
parser.add_argument('--from_checkpoint',
                    type=str,
                    default="none",
                    help='Use checkpoint listed by experiment number'
                    )
parser.add_argument('--stft_small',
                    type=parse_keyword,
                    default=True,
                    metavar='BOOL',
                    help='If [fourier], whether to use a small or large container'
                    )
parser.add_argument('--ft_container',
                    type=str,
                    default='mag',
                    metavar='STR',
                    help='If [fourier], container to use: [mag], [phase], [magphase]'
                    )
parser.add_argument('--thet',
                    type=float,
                    default=1,
                    metavar='DOUBLE',
                    help='Theta hyperparameter (only for magphase)'
                    )
parser.add_argument('--mp_encoder',
                    type=str,
                    default='single',
                    metavar='STR',
                    help='If [fourier] and [magphase], type of magphase encoder: [single], [double]'
                    )
parser.add_argument('--mp_decoder',
                    type=str,
                    default='unet',
                    metavar='STR',
                    help='If [fourier] and [magphase], type of magphase encoder: [unet], [double]'
                    )
parser.add_argument('--mp_join',
                    type=str,
                    default='mean',
                    metavar='STR',
                    help='If [fourier] and [magphase] and [decoder=double], type of join operation: [mean], [2D], [3D]'
                    )
parser.add_argument('--permutation',
                    type=parse_keyword,
                    default=False,
                    metavar='BOOL',
                    help='Permute the encoded image before adding it to the audio'
                    )
parser.add_argument('--embed',
                    type=str,
                    default='stretch',
                    metavar='STR',
                    help='Method of adding the image into the audio: [stretch], [blocks], [blocks2], [blocks3], [multichannel], [luma]'
                    )
parser.add_argument('--luma',
                    type=parse_keyword,
                    default=False,
                    metavar='BOOL',
                    help='Add luma component as the fourth pixelshuffle value'
                    )

# need to design to fit U-net -> fft to (512,160,2)
parser.add_argument('--num_points',
                    type=int,
                    default=16000,
                    help="the length model can handle")
parser.add_argument('--n_fft',
                    type=int,
                    default=1000)
parser.add_argument('--hop_length',
                    type=int,
                    default=400)
parser.add_argument("--mag",
                    type=bool,
                    default=False,
                    help="only use magnitude")
parser.add_argument('--num_layers',
                    type=int,
                    default=2)
parser.add_argument('--transform',
                    type=str,
                    choices=["ID", "TC", "RS", "VC", "WM"],
                    default="ID",
                    )
parser.add_argument("--watermark_len",
                    type=int,
                    default=32)
parser.add_argument("--dataset_i",
                    type=int,
                    choices=[0, 1],
                    default=0)
parser.add_argument("--shift_ratio",
                    type=float,
                    default=0.1
                    )
parser.add_argument("--share_param",
                    type=parse_keyword,
                    default=False,
                    metavar='BOOL')

if __name__ == '__main__':
    set_reproductibility()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    args = parser.parse_args()
    print(args)

    train_loader = loader(
        set='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_points=args.num_points,
        dataset_i=args.dataset_i,
        watermark_len=args.watermark_len,
        shift_ratio=args.shift_ratio
    )
    test_loader = loader(
        set='test',
        batch_size=args.batch_size,
        shuffle=True,
        num_points=args.num_points,
        dataset_i=args.dataset_i,
        watermark_len=args.watermark_len,
        shift_ratio=args.shift_ratio
    )

    print(f"share param: {args.share_param}")
    model = StegoUNet(
        transform=args.transform,
        num_points=args.num_points,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        mag=args.mag,
        num_layers=args.num_layers,
        watermark_len=args.watermark_len,
        shift_ratio=args.shift_ratio,
        share_param=args.share_param
    )
    # if args.transform == "WM":
    #     wm_len = 16
    #     ckpt_path = f"1-multi_IDwl{wm_len}lr1e-4audioMSElam100/30-1-multi_IDwl16lr1e-4audioMSElam100.pt"
    #     state_dict = torch.load(os.path.join(os.environ.get('OUT_PATH'), ckpt_path))["state_dict"]
    #     model.wm_model.load_state_dict(state_dict)

    if args.from_checkpoint.lower() != "none":
        # Load checkpoint
        # checkpoint = torch.load(os.path.join(os.environ.get('OUT_PATH'),f'{args.experiment}-{args.summary}.pt'), map_location='cpu')
        checkpoint = torch.load(os.path.join(os.environ.get('OUT_PATH'), args.from_checkpoint), map_location='cpu')
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        if args.share_param:
            state_dict = {k:v for k, v in checkpoint["state_dict"].items() if "hinet_r" not in k}
        else:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print('Checkpoint loaded')

    print('Ready to train!')

    train(
        model=model,
        tr_loader=train_loader,
        vd_loader=test_loader,
        beta=args.beta,
        lam=args.lam,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        epochs=args.num_epochs,
        val_itvl=args.val_itvl,
        val_size=args.val_size,
        slide=1,
        prev_epoch=checkpoint['epoch'] if args.from_checkpoint.lower() != "none" else None,
        prev_i=checkpoint['i'] if args.from_checkpoint.lower() != "none" else None,
        summary=args.summary,
        experiment=args.experiment,
        transform=args.transform,
        stft_small=args.stft_small,
        ft_container=args.ft_container,
        thet=args.thet,
        dtw=args.dtw
    )
