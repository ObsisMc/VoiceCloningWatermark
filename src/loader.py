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
AUDIO_FOLDER = [f"{DATA_FOLDER}/LibrispeechVoiceClone_",
                f"{DATA_FOLDER}/FSDnoisy/FSDnoisy18k.audio_"]


def preprocess_audio(audio: str | torch.Tensor, num_points, shift_ratio: float, padding="zeros"):
    if isinstance(audio, str):
        sound, sr = torchaudio.load(audio)
    else:
        sound = audio

    device = sound.device
    C, L = sound.shape
    if C > 1:
        raise NotImplemented("Can only handle sounds with over one channel")
    else:
        shift_len_max = int(num_points * shift_ratio)
        total_points = num_points + shift_len_max * 2
        if L < total_points:
            # simplify the situation
            if L < num_points:
                pad_vec = torch.zeros((sound.shape[0], num_points - L)).to(device)
                sound = torch.cat([sound, pad_vec], dim=-1)
            else:
                sound = sound[:, :num_points]

            shift_left = torch.zeros((sound.shape[0], shift_len_max)).to(device)
            shift_right = torch.zeros((sound.shape[0], shift_len_max)).to(device)
        else:
            shift_left = sound[:, :shift_len_max]
            shift_right = sound[:, num_points + shift_len_max: num_points + shift_len_max * 2]
            sound = sound[:, shift_len_max:num_points + shift_len_max]

    sound = sound.squeeze(0).to(device)  # (L,)
    if shift_len_max != 0:
        shift_sound = [shift_left.squeeze(0).to(device), shift_right.squeeze(0).to(device)]
    else:
        shift_sound = []
    return sound, shift_sound


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
            audio_root_i: int,
            folder: str,
            num_points: int,
            watermark_len: int,
            shift_ratio: float
    ):
        self.audio_root_i = audio_root_i
        self.data_root = pathlib.Path(f"{AUDIO_FOLDER[self.audio_root_i]}{folder}")
        self.max_data_num = 50000 if folder == "train" else 3600
        self.num_points = num_points
        self.watermark_len = watermark_len
        self.shift_ratio = shift_ratio

        # dataset 0
        if self.audio_root_i == 0:
            # audio, transcript path
            total_data_num = len(os.listdir(self.data_root))
            self.data_index = torch.randperm(total_data_num)[:self.max_data_num]

            # default audio, transcript and text_prompt
            data_path = os.path.join(self.data_root, "1000")
            self.default_audio, self.shift_sound = preprocess_audio(
                os.path.join(data_path, "speech.wav"),
                num_points=self.num_points,
                shift_ratio=self.shift_ratio)
            with open(os.path.join(data_path, "text.txt"), "r") as f:
                self.default_transcript = self.default_text_prompt = f.read()
            assert self.default_transcript != "" and self.default_text_prompt != ""

        # dataset 1
        elif self.audio_root_i == 1:
            self.audio_names = os.listdir(self.data_root)
            self.data_index = torch.randperm(len(self.audio_names))[:self.max_data_num]

        else:
            raise ValueError(f"Unknown dataset index {self.audio_root_i}")

        print(f'DATA LOCATED AT: {self.data_root}')
        print('Set up done')

    def __len__(self):
        return self.data_index.size(-1)

    def __getitem__(self, index):

        transcript = text_prompt = ""
        if self.audio_root_i == 0:
            # load cloned audio and its transcript
            data_index = self.data_index[index].item()
            data_path = os.path.join(self.data_root, str(data_index))
            audio, shift_sound = preprocess_audio(os.path.join(data_path, "speech.wav"),
                                                  num_points=self.num_points,
                                                  shift_ratio=self.shift_ratio)
            with open(os.path.join(data_path, "text.txt"), "r") as f:
                transcript = f.read()
            if transcript == "":
                audio = self.default_audio
                transcript = self.default_transcript
                shift_sound = self.shift_sound
                print(f"Sample {data_index} has empty text")


            # audio = self.default_audio
            # transcript = self.default_transcript
            # shift_sound = self.shift_sound

            # load text_prompt
            text_prompt_index = torch.randint(len(self), (1,)).item()
            text_prompt_path = os.path.join(self.data_root, str(text_prompt_index))
            with open(os.path.join(text_prompt_path, "text.txt"), "r") as f:
                text_prompt = f.read()
            if text_prompt == "":
                text_prompt = self.default_text_prompt
                print(f"Sample {text_prompt_index} has empty text")
            text_prompt = "Test Voice Cloning"

        elif self.audio_root_i == 1:
            data_index = self.data_index[index].item()
            data_path = os.path.join(self.data_root, self.audio_names[data_index])
            audio, shift_sound = preprocess_audio(os.path.join(data_path),
                                                  num_points=self.num_points,
                                                  shift_ratio=self.shift_ratio)
        else:
            raise ValueError("Unknown dataset")

        # generate watermark
        # torch.manual_seed(rand_seq_seed)
        sequence = torch.rand(self.watermark_len)  # (secret_len,)
        sequence_binary = (sequence > 0.5).int()

        return (sequence, sequence_binary), audio, transcript, text_prompt, shift_sound


def loader(set, num_points, batch_size, shuffle, watermark_len, dataset_i, shift_ratio):
    """
    Prepares the custom dataloader.
    - [set] defines the set type. Can be either [train] or [test].
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    """
    print('Preparing dataset...')

    dataset = StegoDataset(
        audio_root_i=dataset_i,
        folder=set,
        num_points=num_points,
        watermark_len=watermark_len,
        shift_ratio=shift_ratio
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
