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
from loader import loader
from umodel import StegoUNet
from train import train, validate
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
    if keyword.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif keyword.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Wrong keyword.')


parser = argparse.ArgumentParser()
parser.add_argument('--beta',
                        type=float,
                        default=0.5,
                        metavar='DOUBLE',
                        help='Beta hyperparameter'
                    )
parser.add_argument('--lam',
                        type=float,
                        default=1,
                        metavar='DOUBLE',
                        help='Lambda hyperparameter'
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
                        default=1,
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
                        default="Test_MagPhase",
                        metavar='STRING',
                        help='Summary to be shown in wandb'
                    )
parser.add_argument('--from_checkpoint',
                        type=parse_keyword,
                        default=False,
                        metavar='BOOL',
                        help='Use checkpoint listed by experiment number'
                    )
parser.add_argument('--transform',
                        type=str,
                        default='fourier',
                        metavar='STR',
                        help='Which transform to use: [cosine] or [fourier]'
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
                    default=64000 - 400,
                    help="the length model can handle")
parser.add_argument('--n_fft',
                    type=int,
                    default=1022)
parser.add_argument('--hop_length',
                    type=int,
                    default=400)


if __name__ == '__main__':
    set_reproductibility()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    args = parser.parse_args()
    print(args)

    train_loader = loader(
        set='train',
        transform=args.transform,
        stft_small=args.stft_small,
        batch_size=args.batch_size,
        shuffle=True,
        num_points=args.num_points
    )
    test_loader = loader(
        set='test',
        transform=args.transform,
        stft_small=args.stft_small,
        batch_size=args.batch_size,
        shuffle=True,
        num_points=args.num_points
    )

    model = StegoUNet(
        transform=args.transform,
        stft_small=args.stft_small,
        ft_container=args.ft_container,
        mp_encoder=args.mp_encoder,
        mp_decoder=args.mp_decoder,
        mp_join=args.mp_join,
        permutation=args.permutation,
        embed=args.embed,
        luma=args.luma,
        num_points=args.num_points,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )

    if args.from_checkpoint:
        # Load checkpoint
        checkpoint = torch.load(os.path.join(os.environ.get('OUT_PATH'),f'{args.experiment}-{args.summary}.pt'), map_location='cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        print('Checkpoint loaded')

    print('Ready to train!')

    train(
        model=model,
        tr_loader=train_loader,
        vd_loader=test_loader,
        beta=args.beta,
        lam=args.lam,
        lr=args.lr,
        epochs=args.num_epochs,
        val_itvl=args.val_itvl,
        val_size=args.val_size,
        slide=15,
        prev_epoch=checkpoint['epoch'] if args.from_checkpoint else None,
        prev_i=checkpoint['i'] if args.from_checkpoint else None,
        summary=args.summary,
        experiment=args.experiment,
        transform=args.transform,
        stft_small=args.stft_small,
        ft_container=args.ft_container,
        thet=args.thet,
        dtw=args.dtw
    )
