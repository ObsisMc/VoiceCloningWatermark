'''
loader.py

* Image and audio preprocessing
* Data set class
* Data loader
'''

import os
import re
import torch
import random
import pathlib
import torchaudio
import numpy as np
import glob as glob
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
MY_FOLDER = os.environ.get('USER_PATH')
DATA_FOLDER = os.environ.get('DATA_PATH')
AUDIO_FOLDER = f"{DATA_FOLDER}/FSDnoisy/FSDnoisy18k.audio_"
IMAGE_FOLDER = f'{DATA_FOLDER}/imagenet'


def preprocess_audio(audio: str | torch.Tensor, num_points: int = 64000, padding="zeros"):
    if isinstance(audio, str):
        sound, sr = torchaudio.load(audio)
    else:
        sound = audio

    device = sound.device
    C, L = sound.shape
    if C > 1:
        raise NotImplemented("Can only handle sounds with one channel")
    else:
        if L < num_points:
            # TODO: padding zeros or others like gaussian noise
            if padding == "zeros":
                pad_vec = torch.zeros((sound.shape[0], num_points - L)).to(device)
            elif padding == "gaussian":
                pad_vec = torch.normal(0, 1e-4, (sound.shape[0], num_points - L)).to(device)
            else:
                raise ValueError("Invalid value of padding of preprocess_audio()")
            sound = torch.cat([sound, pad_vec], dim=-1)
        else:
            sound = sound[:, :num_points]

    sound = sound.squeeze(0)  # (L,)
    return sound.to(device)


class StegoDataset(torch.utils.data.Dataset):
    """
    Custom datasets pairing images with spectrograms.
    - [image_root] defines the path to read the images from.
    - [audio_root] defines the path to read the audio clips from.
    - [folder] can be either [train] or [test].
    - [mappings] is the dictionary containing a descriptive name for 
    the images from ImageNet. It is used to index the different
    subdirectories.
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    - [image_extension] defines the extension of the image files. 
    By default it is set to JPEG.
    - [audio_extension] defines the extension of the audio files. 
    By default it is set to WAV.
    """

    def __init__(
            self,
            image_root: str,
            audio_root: str,
            folder: str,
            mappings: dict,
            num_points: int,
            rgb: bool = True,
            transform: str = 'cosine',
            stft_small: bool = True,
            image_extension: str = "JPEG",
            audio_extension: str = "wav"
    ):

        # self._image_data_path = pathlib.Path(image_root) / folder
        self._image_data_path = pathlib.Path(image_root) / 'train'
        self._audio_data_path = pathlib.Path(f'{audio_root}{folder}')
        self._MAX_LIMIT = 10000 if folder == 'train' else 900
        self._TOTAL = 10000
        self._MAX_AUDIO_LIMIT = 17584 if folder == 'train' else 946
        self._colorspace = 'RGB' if rgb else 'L'
        self._transform = transform
        self._stft_small = stft_small
        self._num_points = num_points

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')
        print(f'AUDIO DATA LOCATED AT: {self._audio_data_path}')

        self.image_extension = image_extension
        self.audio_extension = audio_extension
        self._index = 0
        self._indices = []
        self._audios = []

        # secrets
        self._indices = [i for i in range(self._MAX_LIMIT)]  # seed for generating random vectors
        self._index = self._MAX_LIMIT

        # AUDIO PATH RETRIEVING (here the paths for test and train are different)
        self._index_aud = 0

        for audio_path in glob.glob(f'{self._audio_data_path}/*.{self.audio_extension}'):

            self._audios.append(audio_path)
            self._index_aud += 1

            if (self._index_aud == self._MAX_AUDIO_LIMIT): break

        # self._AUDIO_PROCESSOR = AudioProcessor(transform=self._transform, stft_small=self._stft_small)

        print('Set up done')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        rand_seq_seed = self._indices[index]
        rand_indexer = random.randint(0, self._MAX_AUDIO_LIMIT - 1)

        audio_path = self._audios[rand_indexer]

        # torch.manual_seed(rand_seq_seed)
        sequence = torch.rand(32)  # (secret_len,)
        # TODO obsismc: this may generated a sequence with regular pattern, which may be easy to learn or hack
        sequence_binary = (sequence > 0.5).int()

        # magnitude_stft, phase_stft = self._AUDIO_PROCESSOR.forward(audio_path)
        audio = preprocess_audio(audio_path, num_points=self._num_points)  # (L,)
        return (sequence, sequence_binary), audio


def loader(set='train', num_points=64000, rgb=True, transform='cosine', stft_small=True, batch_size=1, shuffle=False):
    """
    Prepares the custom dataloader.
    - [set] defines the set type. Can be either [train] or [test].
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    """
    print('Preparing dataset...')
    mappings = {}
    with open(f'{IMAGE_FOLDER}/mappings.txt') as f:
        for line in f:
            words = line.split()
            mappings[words[0]] = words[1]

    dataset = StegoDataset(
        image_root=f'{IMAGE_FOLDER}/ILSVRC/Data/CLS-LOC',
        audio_root=AUDIO_FOLDER,
        folder=set,
        mappings=mappings,
        rgb=rgb,
        transform=transform,
        stft_small=stft_small,
        num_points=num_points
    )

    print('Dataset prepared.')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle
    )

    print('Data loaded ++')
    return dataloader
