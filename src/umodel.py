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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.8, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.8, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, downsample_factor=4):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(downsample_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, opp_channels=-1):
        # opp_channels -> The number of channels (depth) of the opposite replica of the unet
        #                   If -1, the same number as the current image is assumed
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=0),
            nn.LeakyReLU(0.8, inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, output_padding=0),
            nn.LeakyReLU(0.8, inplace=True),
        )
        if opp_channels == -1:
            opp_channels = out_channels
        self.conv = DoubleConv(opp_channels + out_channels, out_channels)

    def forward(self, mix, im_opposite, au_opposite=None):
        mix = self.up(mix)
        # print(f"up layer: {mix.shape}")
        x = torch.cat((mix, im_opposite), dim=1)
        return self.conv(x)


class PrepHidingNet(nn.Module):
    def __init__(self, transform='cosine', stft_small=True, embed='stretch', secrete_len=32, num_points=64000,
                 n_fft=1000, hop_length=400, mag=False):
        super(PrepHidingNet, self).__init__()
        self._transform = transform
        self._stft_small = stft_small
        self.embed = embed
        self._secrete_len = secrete_len
        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag = mag

        self.fc = nn.Linear(self._secrete_len, self.num_points)  # TODO: may be optimized
        # self.im_encoder_layers = nn.ModuleList([
        #     Down(1 + (not self.mag), 64),
        #     Down(64, 64 * 2)
        # ])
        # self.im_decoder_layers = nn.ModuleList([
        #     Up(64 * 2, 64),
        #     Up(64, 1 + (not self.mag))
        # ])
        self.subnet = ResidualDenseBlock_out(2, 2)

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        return tmp

    def forward(self, seq):
        # TODO obsismc: the batch size must be 1
        seq = self.fc(seq)  # (B,L)
        seq_fft_img = self.stft(seq)
        seq_fft_real = torch.view_as_real(seq_fft_img)  # (B,N,T,C)
        phase = None
        if self.mag:
            mag, phase = spec2magphase(seq_fft_real)
            seq_fft_real = mag
        seq_fft_real = seq_fft_real.permute(0, 3, 1, 2)  # (B,C,N,T)

        # seq_wavs_down = [seq_fft_real]
        # # Encoder part of the UNet
        # for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
        #     seq_wavs_down.append(enc_layer(seq_wavs_down[-1]))
        #
        # seq_wavs_up = [seq_wavs_down.pop()]
        #
        # # Decoder part of the UNet
        # for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
        #     seq_wavs_up.append(dec_layer(seq_wavs_up[-1], seq_wavs_down[-1 - dec_layer_idx], None))

        seq_fft_ret = self.subnet(seq_fft_real)

        return seq_fft_ret.permute(0, 2, 3, 1), phase  # (B,N,T,C), (B,N,T,1)


class RevealNet(nn.Module):
    def __init__(self, mp_decoder=None, stft_small=True, embed='stretch', luma=False, secrete_len=32,
                 transform='fourier',
                 num_points=64000,
                 n_fft=1000,
                 hop_length=400,
                 mag=False):
        super(RevealNet, self).__init__()

        self.mp_decoder = mp_decoder
        self._stft_small = stft_small
        self.embed = embed
        self.luma = luma

        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag = mag

        # self.im_encoder_layers = nn.ModuleList([
        #     Down(1 + (not self.mag), 64),
        #     Down(64, 64 * 2)
        # ])
        # self.im_decoder_layers = nn.ModuleList([
        #     Up(64 * 2, 64),
        #     Up(64, 1 + (not self.mag))
        # ])
        self.subnet = ResidualDenseBlock_out(2, 2)
        self.fc = nn.Linear(self.num_points, secrete_len)

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def forward(self, ct, phase=None):
        ct = ct.permute(0, 3, 1, 2)  # (B,C,N,T)
        # ct_down = [ct]
        #
        # # Encoder part of the UNet
        # for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
        #     ct_down.append(enc_layer(ct_down[-1]))
        #
        # ct_up = [ct_down.pop(-1)]
        #
        # # Decoder part of the UNet
        # for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
        #     ct_up.append(
        #         dec_layer(ct_up[-1],
        #                   ct_down[-1 - dec_layer_idx])
        #     )
        #
        # revealed = ct_up[-1]

        revealed = self.subnet(ct)

        # obsismc: sequence
        revealed = revealed.permute(0, 2, 3, 1).contiguous()  # (B,N,T,C)
        if self.mag:
            spec = magphase2spec(revealed, phase)
            spec_img = torch.view_as_complex(spec.contiguous())
        else:
            spec_img = torch.view_as_complex(revealed)
        revealed = self.istft(spec_img)
        revealed = self.fc(revealed)
        revealed = torch.sigmoid(revealed)

        return revealed  # (B,secret_len)


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
            #wav = torch.from_numpy(
            #    librosa.resample(
            #        wav.detach().cpu().numpy(), orig_sr=num_points, target_sr=int(num_points * alpha.item())
            #    )
            #).float().to(wav.device)
            # wav = torchaudio.functional.resample(wav, num_points, int(num_points * alpha.item()))
            wav =torchaudio.transforms.Resample(num_points, int(num_points * alpha.item())).to(device)(wav)
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
                 mag, num_layers, watermark_len, shift_ratio):
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

        # wavmark
        self.watermark_fc = nn.Linear(self.watermark_len, self.num_points)
        self.hinet = Hinet(num_layers=self.num_layers)
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

    def forward(self, secret, cover, transcripts, text_prompts, shift_sound):
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
                transform_ct_wav_tmp = self.transform_layer(watermark_ct_wav[i][None,...],
                                                           {"sample_rate": self.sr,
                                                            "transcript": transcript,
                                                            "text_prompt": text_prompt})
                transform_ct_wavs.append(transform_ct_wav_tmp)
            transform_ct_wav = torch.cat(transform_ct_wavs, dim=0).to(watermark_ct_wav.device)

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
                transform_ct_wav = torch.cat([shift_sound[:, self.shift_max_len-shift_len:], transform_ct_wav[:, :self.shift_max_len-shift_len]], dim=-1).to(
                    transform_ct_wav.device)

        
        ## length alignment  TODO: handle unfixed length
        # transform_ct_wav = self.align(transform_ct_wav)

        ## stft before decode
        signal_fft = self.stft(transform_ct_wav)
        sign_fft_real = torch.view_as_real(signal_fft)

        ## direct connect
        # sign_fft_real = signal_wmd_fft_real
        

        watermark_fft_real = sign_fft_real  # obsismc: different from what the paper says
        _, message_restored_fft_real = self.enc_dec(sign_fft_real, watermark_fft_real, rev=True)
        message_restored_fft = torch.view_as_complex(message_restored_fft_real.contiguous())
        message_restored_expanded = self.istft(message_restored_fft)
        revealed = self.watermark_fc_back(message_restored_expanded)
        revealed = torch.sigmoid(revealed)

        return_ct_fft = cover_fft_real

        return cover_fft_real, return_ct_fft, watermark_ct_wav, revealed

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
        _, message_restored_fft_real = self.enc_dec(sign_fft_real, watermark_fft_real, rev=True)
        message_restored_fft = torch.view_as_complex(message_restored_fft_real.contiguous())
        message_restored_expanded = self.istft(message_restored_fft)
        revealed = self.watermark_fc_back(message_restored_expanded)
        revealed = torch.sigmoid(revealed)
        return revealed


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
