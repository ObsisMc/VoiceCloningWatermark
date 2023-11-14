'''
umodel.py

* Auxiliary functions:
    - RGB to YCbCr
    - Pixel unshuffle
    - Convolution layers
* Subnets:
    - PrepHidingNet
    - RevealNet
* Main net: StegoUNet
'''

import os
import torch
import librosa
import numpy as np
import torch.nn as nn
from torch import utils
import torchaudio
import torch.nn.functional as F
from src.loader import preprocess_audio
from src.rrdb_denselayer import ResidualDenseBlock_out
from src.hinet import Hinet

from utils.prompt_making import make_prompt_train, make_prompt
from utils.generation import SAMPLE_RATE, generate_audio_train, preload_models, generate_audio


def stft(self, data):
    window = torch.hann_window(self.n_fft).to(data.device)
    tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
    # [1, 501, 41, 2]
    return tmp


def istft(data, n_fft, hop_length):
    window = torch.hann_window(n_fft).to(data.device)
    return torch.istft(data, n_fft=n_fft, hop_length=hop_length, window=window,
                       return_complex=False)


def spec2magphase(spec: torch.Tensor):
    rea = spec[..., 0]
    imag = spec[..., 1]
    mag = torch.sqrt(rea ** 2 + imag ** 2 + 1e-5).to(spec.device).unsqueeze(-1)
    phase = torch.atan2(imag, rea).to(spec.device).unsqueeze(-1)
    return mag, phase


def magphase2spec(mag: torch.Tensor, phase: torch.Tensor):
    r = mag * torch.cos(phase)
    i = mag * torch.sin(phase)
    return torch.cat([r, i], dim=-1).to(mag.device)


class Transform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wav, num_points, name, data_dict):
        wav_l = wav.size(1)
        device = wav.device
        if name == "ID":
            pass
        elif name == "TC":  # truncate
            alpha = torch.randint(3, 10, (1,))
            wav = wav[:, :int(num_points * alpha / 10)]
        elif name == "RS":  # resample to make it longer
            alpha = torch.rand(1) * 1. + 0.5
            # wav = torch.from_numpy(
            #    librosa.resample(
            #        wav.detach().cpu().numpy(), orig_sr=num_points, target_sr=int(num_points * alpha.item())
            #    )
            # ).float().to(wav.device)
            # wav = torchaudio.functional.resample(wav, num_points, int(num_points * alpha.item()))
            wav = torchaudio.transforms.Resample(num_points, int(num_points * alpha.item())).to(device)(wav)
        # TODO: add noise
        elif name == "NS":
            noise = torch.normal(0, 1e-4, (wav.size(0), wav.size(1))).to(device)
            wav += noise
        elif name == "VC":
            sr, transcript, text_prompt = data_dict["sample_rate"], data_dict["transcript"], data_dict["text_prompt"]

            with torch.no_grad():
                audio_tokens, text_tokens, lang_pr = make_prompt_train(name="clone",
                                                                       audio_prompt=wav,
                                                                       sr=sr,
                                                                       transcript=transcript)
                wav = generate_audio_train(text_prompt,
                                           audio_tokens=audio_tokens,
                                           text_tokens=text_tokens, lang_pr=lang_pr)

            wav = wav.detach().clone().unsqueeze(0).to(device)
        elif name == "WM":
            wm_model, wm_len = data_dict["model"], data_dict["watermark_len"]
            wm = torch.rand(wm_len).repeat(wav.size(0), 1).to(device)
            with torch.no_grad():
                wav_encode = wm_model.encode(wm, wav)
                _, wav_decode = wm_model.decode(wav_encode)
            wav = wav_decode
        else:
            raise ValueError("Invalid transform name")

        ctx.save_for_backward(wav, torch.tensor(wav_l))
        if wav.shape[1] < wav_l:
            wav = F.pad(wav, (0, wav_l - wav.shape[1]), "constant", 0)
        if wav.shape[1] > wav_l:
            wav = wav[:, :wav_l]
        return wav

    @staticmethod
    def backward(ctx, grad_outputs):
        wav, wav_l = ctx.saved_tensors
        return grad_outputs[:wav_l], None, None, None


class StegoUNet(nn.Module):
    def __init__(self, transform, num_points, n_fft, hop_length,
                 mag, num_layers, watermark_len, shift_ratio, share_param):
        assert mag == False and num_layers > 0

        super().__init__()

        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag = mag
        self.num_layers = num_layers
        self.sr = 16000
        self.transform = transform
        self.watermark_len = watermark_len
        self.shift_ratio = shift_ratio
        self.shift_max_len = int(self.num_points * self.shift_ratio)
        self.share_param = share_param

        # wavmark
        self.watermark_fc = nn.Linear(self.watermark_len, self.num_points)
        self.hinet = Hinet(num_layers=self.num_layers)
        if self.share_param:
            self.hinet_r = self.hinet
        else:
            self.hinet_r = Hinet(num_layers=self.num_layers)
        self.watermark_fc_back = nn.Linear(self.num_points, self.watermark_len)

        # alignment
        self.align = AlignmentLayer(self.num_points)

        # transform
        self.transform_layer = lambda audio, data_dict: Transform.apply(audio, self.num_points, self.transform,
                                                                        data_dict)
        if self.transform == "VC":
            print("Loading Voice Cloning model...")
            preload_models()
            print("Finish loading!")

    def stft(self, data, return_complex=True):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                         return_complex=return_complex)
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def forward(self, secret, cover, transcripts, text_prompts, shift_sound, **kwargs):
        # wavmark

        ## encode
        cover_fft = self.stft(cover)
        cover_fft_real = torch.view_as_real(cover_fft)
        # (batch,freq_bins,time_frames,2)

        secret_expand = self.watermark_fc(secret)
        secret_fft = self.stft(secret_expand)
        secret_fft_real = torch.view_as_real(secret_fft)

        signal_wmd_fft_real, msg_remain = self.enc_dec(cover_fft_real, secret_fft_real, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd_fft = torch.view_as_complex(signal_wmd_fft_real.contiguous())
        watermark_ct_wav = self.istft(signal_wmd_fft)

        ## transform
        if self.transform == "VC":
            transform_ct_wavs = []
            for i, (transcript, text_prompt) in enumerate(zip(transcripts, text_prompts)):
                # print(f"transcript: {transcript} <---> text_prompt: {text_prompt}")
                transform_ct_wav_tmp = self.transform_layer(watermark_ct_wav[i][None, ...],
                                                            {"sample_rate": self.sr,
                                                             "transcript": transcript,
                                                             "text_prompt": text_prompt})
                transform_ct_wavs.append(transform_ct_wav_tmp)
            transform_ct_wav = torch.cat(transform_ct_wavs, dim=0).to(watermark_ct_wav.device)
        elif self.transform == "WM":
            transform_ct_wav = self.transform_layer(watermark_ct_wav, {"model": kwargs["wm_model"],
                                                                       "watermark_len": self.watermark_len})
        else:
            transform_ct_wav = self.transform_layer(watermark_ct_wav, None)

        ## shift
        if shift_sound:
            shift_idx = 1
            shift_sound = shift_sound[shift_idx]
            shift_len = int(self.num_points * np.random.uniform(0, self.shift_ratio))

            if shift_idx:
                transform_ct_wav = torch.cat([transform_ct_wav[:, shift_len:], shift_sound[:, :shift_len]], dim=-1).to(
                    transform_ct_wav.device)
            else:
                transform_ct_wav = torch.cat([shift_sound[:, self.shift_max_len - shift_len:],
                                              transform_ct_wav[:, :self.shift_max_len - shift_len]], dim=-1).to(
                    transform_ct_wav.device)

        ## length alignment  TODO: handle unfixed length
        # transform_ct_wav = self.align(transform_ct_wav)

        ## stft before decode
        signal_fft = self.stft(transform_ct_wav)
        sign_fft_real = torch.view_as_real(signal_fft)

        ## direct connect
        # sign_fft_real = signal_wmd_fft_real

        # reveal watermark and original audio
        watermark_fft_real = sign_fft_real  # obsismc: different from what the paper says
        audio_restored_fft_real, message_restored_fft_real = self.enc_dec(sign_fft_real, watermark_fft_real, rev=True)

        message_restored_fft = torch.view_as_complex(message_restored_fft_real.contiguous())
        message_restored_expanded = self.istft(message_restored_fft)
        revealed = self.watermark_fc_back(message_restored_expanded)
        revealed = torch.sigmoid(revealed)

        audio_restored_fft = torch.view_as_complex(audio_restored_fft_real.contiguous())
        audio_restored = self.istft(audio_restored_fft)

        # useless
        return_ct_fft = cover_fft_real

        return cover_fft_real, return_ct_fft, watermark_ct_wav, revealed, audio_restored

    def enc_dec(self, signal, watermark, rev):
        signal = signal.permute(0, 3, 2, 1)
        # [4, 2, 41, 501]
        watermark = watermark.permute(0, 3, 2, 1)
        if not rev:
            signal2, watermark2 = self.hinet(signal, watermark, rev)
        else:
            signal2, watermark2 = self.hinet_r(signal, watermark, rev)
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)

    def encode(self, secret, cover):
        cover_fft = self.stft(cover)
        cover_fft_real = torch.view_as_real(cover_fft)
        # (batch,freq_bins,time_frames,2)

        secret_expand = self.watermark_fc(secret)
        secret_fft = self.stft(secret_expand)
        secret_fft_real = torch.view_as_real(secret_fft)

        signal_wmd_fft_real, msg_remain = self.enc_dec(cover_fft_real, secret_fft_real, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd_fft = torch.view_as_complex(signal_wmd_fft_real.contiguous())
        watermark_ct_wav = self.istft(signal_wmd_fft)
        return watermark_ct_wav

    def decode(self, audio):
        signal_fft = self.stft(audio)
        sign_fft_real = torch.view_as_real(signal_fft)
        watermark_fft_real = sign_fft_real  # obsismc: different from what the paper says
        audio_restored_fft_real, message_restored_fft_real = self.enc_dec(sign_fft_real, watermark_fft_real, rev=True)

        # restored watermark
        message_restored_fft = torch.view_as_complex(message_restored_fft_real.contiguous())
        message_restored_expanded = self.istft(message_restored_fft)
        revealed = self.watermark_fc_back(message_restored_expanded)
        revealed = torch.sigmoid(revealed)

        # restored audio
        audio_restored_fft = torch.view_as_complex(audio_restored_fft_real.contiguous())
        audio_restored = self.istft(audio_restored_fft)

        return revealed, audio_restored


class AlignmentLayer(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

        self.avg = nn.AdaptiveAvgPool1d(self.num_points)
        self.mx = nn.AdaptiveMaxPool1d(self.num_points)
        self.conv = nn.Conv1d(3, 1, 3, 1, 1)
        # self.conv1 = nn.Conv1d(3, 16, 3, 1, 1)
        # self.conv2 = nn.Conv1d(3 + 16, 1, 3, 1, 1)
        # self.conv3 = nn.Conv1d(3 + 2 * 16, 1, 3, 1, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        B, L = x.size()

        x_avg = self.avg(x)
        x_mx = self.mx(x)
        x = torch.stack([x[:, :self.num_points], x_avg, x_mx], dim=1)
        x = self.conv(x)
        # x1 = self.act(self.conv1(x))
        # x2 = self.conv2(torch.cat([x, x1], dim=1))
        # x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x = x.squeeze(1)
        return x


class VoiceCloner:
    def __init__(self):
        preload_models()

    def clone(self, audio_prompt_path: str, text_prompt: str, transcript=None):
        if transcript is not None:
            make_prompt(name="clone", audio_prompt_path=audio_prompt_path, transcript=transcript)
        else:
            make_prompt(name="clone", audio_prompt_path=audio_prompt_path)

        # TODO: return numpy, which means cannot backprop gradient
        audio_array = generate_audio(text_prompt, prompt="clone")
        audio_array = torch.tensor(audio_array).unsqueeze(0)
        return audio_array, SAMPLE_RATE
