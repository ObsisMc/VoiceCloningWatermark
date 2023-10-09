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


class AudioProcessor():
    """
    Function to preprocess the audios from the custom 
    dataset. We set the [_limit] in terms of samples,
    the [_frame_length] and [_frame_step] of the [transform]
    transform. 

    If transform is [cosine] it returns just the STDCT matrix.
    Else, if transform is [fourier] returns the STFT magnitude
    and phase.
    """

    def __init__(self, transform, stft_small=True, random_init=True):
        # Corresponds to 1.5 seconds approximately
        if transform == 'cosine':
            self._frame_length = 2 ** 10
            self._frame_step = 2 ** 7 + 2
        else:
            if stft_small:
                self._frame_length = 2 ** 11 - 1
                self._frame_step = 2 ** 7 + 4
            else:
                self._frame_length = 2 ** 12 - 1
                self._frame_step = 2 ** 6 + 2

        self.random_init = random_init

        self._transform = transform
        if self._transform == 'fourier':
            self.stft = STFT(
                filter_length=self._frame_length,
                hop_length=self._frame_step,
                win_length=self._frame_length,
                window='hann'
            )

    @property
    def _limit(self):
        return 67522  # 2 ** 16 + 2 ** 11 - 2 ** 6 + 2

    def get_frame_length(self):
        return self._frame_length

    def get_frame_step(self):
        return self._frame_step

    def forward(self, audio_path, path=True):
        if path:
            self.sound, self.sr = torchaudio.load(audio_path)

            # Get the samples dimension
            sound = self.sound[0]
        else:
            sound = audio_path[0]

        # Create a temporary array
        tmp = torch.zeros([self._limit, ])

        # Check if the audio is shorter than the limit
        if sound.numel() < self._limit:
            # Zero-pad at the end, or randomly at both start and end
            if self.random_init:
                i = random.randint(0, self._limit - len(sound))
                tmp[i:i + sound.numel()] = sound[:]
            else:
                tmp[:sound.numel()] = sound[:]
        else:
            # Use only part of the audio. Either start at beginning or random
            if self.random_init:
                i = random.randint(0, len(sound) - self._limit)
            else:
                i = 0
            tmp[:] = sound[i:i + self._limit]

        if self._transform == 'cosine':
            return sdct_torch(
                tmp.type(torch.float32),
                frame_length=self._frame_length,
                frame_step=self._frame_step
            )
        elif self._transform == 'fourier':
            magnitude, phase = self.stft.transform(tmp.unsqueeze(0).type(torch.float32))
            return magnitude, phase
        else:
            raise Exception(f'Transform not implemented')


def preprocess_audio(audio: str | torch.Tensor, num_points: int = 64000):
    if isinstance(audio, str):
        sound, sr = torchaudio.load(audio)
    else:
        sound = audio

    C, L = sound.shape
    if C > 1:
        raise NotImplemented("Can only handle sounds with one channel")
    else:
        if L < num_points:
            # TODO: padding zeros or others like gaussian noise
            sound = torch.cat([sound, torch.zeros((sound.shape[0], num_points - L))], dim=1)
        else:
            sound = sound[:, :num_points]

    sound = sound.squeeze(0)  # (L,)
    return sound


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
