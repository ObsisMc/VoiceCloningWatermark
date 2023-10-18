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
AUDIO_FOLDER = f"{DATA_FOLDER}/LibrispeechVoiceClone_"


def preprocess_audio(audio: str | torch.Tensor, num_points: int = 64000, padding="zeros"):
    if isinstance(audio, str):
        sound, sr = torchaudio.load(audio)
    else:
        sound = audio

    device = sound.device
    C, L = sound.shape
    if C > 1:
        raise NotImplemented("Can only handle sounds with over one channel")
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

        self.data_root = pathlib.Path(f"{audio_root}{folder}")
        self.max_data_num = 20000 if folder == "train" else 1000
        self.num_points = num_points
        self.watermark_len = 32

        # audio, transcript path
        total_data_num = len(os.listdir(self.data_root))
        torch.manual_seed(2023)
        self.data_index = torch.randperm(total_data_num)[:self.max_data_num]

        print(f'DATA LOCATED AT: {self.data_root}')
        print('Set up done')

    def __len__(self):
        return self.data_index.size(-1)

    def __getitem__(self, index):
        data_index = self.data_index[index].item()
        data_path = os.path.join(self.data_root, str(data_index))

        audio = preprocess_audio(os.path.join(data_path, "speech.wav"), num_points=self.num_points)
        with open(os.path.join(data_path, "text.txt"), "r") as f:
            transcript = f.read()
        text_prompt = "Voice cloning test"

        # torch.manual_seed(rand_seq_seed)
        sequence = torch.rand(self.watermark_len)  # (secret_len,)
        sequence_binary = (sequence > 0.5).int()

        return (sequence, sequence_binary), audio, transcript, text_prompt


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

    dataset = StegoDataset(
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
