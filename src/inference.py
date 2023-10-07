import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from loader import AudioProcessor
from umodel import StegoUNet
import torchaudio
from dotenv import load_dotenv
from losses import calc_ber, signal_noise_ratio
from torch_stft import STFT

load_dotenv()


def parse_keyword(keyword):
    if isinstance(keyword, bool): return keyword
    if keyword.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif keyword.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong keyword.')


parser = argparse.ArgumentParser()

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
pattern = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
           1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
pattern = [0, 1, 1, 1, 1, 0, 1, 1, 0 ,1, 1, 1, 1, 1, 1, 1, 1 ,1, 0, 0, 1,
           0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    args = parser.parse_args()
    print(args)

    model = StegoUNet(
        transform=args.transform,
        stft_small=args.stft_small,
        ft_container=args.ft_container,
        mp_encoder=args.mp_encoder,
        mp_decoder=args.mp_decoder,
        mp_join=args.mp_join,
        permutation=args.permutation,
        embed=args.embed,
        luma=args.luma
    )

    # Load checkpoint
    ckpt_path = '1-Seqence_0923/3-1-Seqence_0923.pt'
    checkpoint = torch.load(os.path.join(os.environ.get('OUT_PATH'), ckpt_path),
                            map_location='cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    print('Checkpoint loaded')

    # load testing data
    audio_name = "example1.wav"
    audio_root = os.path.join(os.environ.get('USER_PATH'), 'test_audio')
    audio_path = os.path.join(audio_root, audio_name)
    audio_path = "/home/zrh/Repository/gitrepo/AudioMNIST/data/17/4_17_6.wav"
    sound, sr = torchaudio.load(audio_path)  # (1,len)
    print(f"Covers' len: {sound.shape}")

    processor = AudioProcessor(args.transform, args.stft_small, False)
    mag, phase = processor.forward(sound, path=False)
    mag = mag.unsqueeze(0)

    # secret  TODO solve out-of-distribution secret
    secret = torch.normal(0.4, 0.2, (32,))
    # secret = torch.rand(32)
    # secret = torch.tensor(pattern[:32]).float()
    # sigmoid
    # secret = 1 / (1 + torch.e**(secret - 0.5))

    secret_binary = (secret > 0.5).int()
    secret, secret_binary = secret.unsqueeze(0), secret_binary.unsqueeze(0)

    # run model
    # TODO add noise
    with torch.no_grad():
        container, revealed = model(secret, mag)

    # stft
    stft = STFT(
        filter_length=2 ** 11 - 1 if args.stft_small else 2 ** 12 - 1,
        hop_length=132 if args.stft_small else 66,
        win_length=2 ** 11 - 1 if args.stft_small else 2 ** 12 - 1,
        window='hann'
    )
    stft.num_samples = 67522
    container_wav = stft.inverse(container.squeeze(1), phase.squeeze(1))

    # evaluate
    # sound, container_wav = sound.detach().numpy(), container_wav.detach().numpy()
    sound_np, container_wav_np = sound.numpy(), container_wav.numpy()
    snr = signal_noise_ratio(sound_np, container_wav_np)
    ber = calc_ber(revealed, secret)

    # TODO SNR is very high but ber isn't good
    print(f"SNR: {snr}, BER: {ber * 100}%")
    print(f"Secret: {(secret > 0.5).int()} (the number of 1: {(secret > 0.5).sum()}/{len(secret[0])})")
    print(f"Revealed: {(revealed > 0.5).int()}")

    # save container_wav
    # torchaudio.save(f"contained_{audio_name}", container_wav, sr)
