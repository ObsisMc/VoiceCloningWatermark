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
import numpy as np
import torch.nn as nn
from torch import utils
import torchaudio
import torch.nn.functional as F
from src.loader import preprocess_audio
from src.rrdb_denselayer import ResidualDenseBlock_out
from src.hinet import Hinet

try:
    from utils.prompt_making import make_prompt
    from utils.generation import SAMPLE_RATE, generate_audio, preload_models

    voice_clone_valid = True
except:
    voice_clone_valid = False
    print("\033[31mCannot Use Voice Cloning!\033[0m")


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
            alpha = torch.rand(1) * 4 + 1
            wav = torchaudio.functional.resample(wav, num_points, int(num_points * alpha.item()))
        # TODO: add noise
        elif name == "NS":
            noise = torch.normal(0, 1e-4, (wav.size(0), wav.size(1))).to(device)
            wav += noise
        elif name == "VC":
            audio_prompt_path, transcript = data_dict["audio_prompt_path"], data_dict["transcript"]
            text_prompt = data_dict["text_prompt"]

            make_prompt(name="clone", audio_prompt_path=audio_prompt_path, transcript=transcript)

            wav = generate_audio(text_prompt, prompt="clone")
            wav = torch.tensor(wav).unsqueeze(0).to(device)
        else:
            raise ValueError("Invalid transform name")

        ctx.save_for_backward(wav, torch.tensor(wav_l))
        wav = preprocess_audio(wav, num_points).unsqueeze(0)
        return wav

    @staticmethod
    def backward(ctx, grad_outputs):
        wav, wav_l = ctx.saved_tensors
        return grad_outputs[:wav_l], None, None, None


class StegoUNet(nn.Module):
    def __init__(self, transform='cosine', stft_small=True, ft_container='mag', mp_encoder='single',
                 mp_decoder='double', mp_join='mean', permutation=False, embed='stretch', luma='luma',
                 num_points=63600, n_fft=1022, hop_length=400, mag=False, num_layers=1):
        assert mag == False and num_layers > 0
        global voice_clone_valid

        super().__init__()

        self.transform = transform
        self.stft_small = stft_small
        self.ft_container = ft_container
        self.mp_encoder = mp_encoder
        self.mp_decoder = mp_decoder
        self.mp_join = mp_join
        self.permutation = permutation
        self.embed = embed
        self.luma = luma

        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag = mag
        self.num_layers = num_layers

        if self.ft_container == 'magphase' and self.embed != 'stretch':
            raise Exception('Mag+phase does not work with embeddings other than stretch')
        if self.luma and self.embed == 'multichannel':
            raise Exception('Luma is not compatible with multichannel')

        if transform != 'fourier' or ft_container != 'magphase':
            self.mp_decoder = None  # For compatiblity with RevealNet

        # Sub-networks
        self.PHN = PrepHidingNet(self.transform, self.stft_small, self.embed, num_points=self.num_points,
                                 n_fft=self.n_fft, hop_length=self.hop_length, mag=self.mag)
        self.encode_subnets = nn.ModuleList([ResidualDenseBlock_out(2, 2) for _ in range(self.num_layers)])

        self.decode_subnets = nn.ModuleList([ResidualDenseBlock_out(2, 2) for _ in range(self.num_layers)])
        self.RN = RevealNet(self.mp_decoder, self.stft_small, self.embed, self.luma, num_points=self.num_points,
                            n_fft=self.n_fft, hop_length=self.hop_length, mag=self.mag)

        # wavmark
        self.watermark_fc = nn.Linear(32, self.num_points)
        self.hinet = Hinet(num_layers=self.num_layers)
        self.hinet_r = Hinet(num_layers=self.num_layers)
        self.watermark_fc_back = nn.Linear(self.num_points, 32)

        # transform
        if voice_clone_valid:
            preload_models()

    def stft(self, data, return_complex=True):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                         return_complex=return_complex)
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def forward(self, secret, cover, cover_phase=None):
        # # Encode the image using PHN
        # hidden_signal, hidden_phase = self.PHN(secret)  # (B,N,T,C)
        #
        # # Residual connection
        # # Also keep a copy of the unpermuted containers to compute the loss
        # cover_fft_img = self.stft(cover)
        # cover_fft_real = torch.view_as_real(cover_fft_img)  # (B,N,T,2)
        # if self.mag:
        #     mag, phase = spec2magphase(cover_fft_real)
        #     cover_fft_real = mag
        # hidden_signal = hidden_signal.permute(0, 3, 1, 2)
        # # container_fft = cover_fft_real + hidden_signal  # (B,N,T,C)
        #
        # # deep encoding
        # container_fft = cover_fft_real.permute(0, 3, 1, 2)  # (B,C,N,T)
        # for encode_subnet in self.encode_subnets:
        #     container_fft = encode_subnet(container_fft + hidden_signal)  # residual
        # container_fft = container_fft.permute(0, 2, 3, 1)  # (B,N,T,C)
        #
        # return_ct_fft = container_fft  # (B,N,T,C)
        # if self.mag:
        #     spec = magphase2spec(container_fft, phase)
        #     spec_img = torch.view_as_complex(spec.contiguous())
        # else:
        #     spec_img = torch.view_as_complex(container_fft.contiguous())
        # watermark_ct_wav = self.istft(spec_img)  # (B,L)
        #
        # # transform
        # # TODO stft won't give out the same spectogram
        # trans_ct_wav = Transform.apply(watermark_ct_wav, self.num_points, "ID", None)
        # # trans_ct_wav = preprocess_audio(watermark_ct_wav, self.num_points).unsqueeze(0)
        # # trans_ct_wav = watermark_ct_wav
        #
        # trans_ct_fft = self.stft(trans_ct_wav)
        # trans_ct_fft_real = torch.view_as_real(trans_ct_fft)
        # if self.mag:
        #     mag_tf, phase_tf = spec2magphase(trans_ct_fft_real)
        #     trans_ct_fft_real = mag_tf
        #
        # # deep decoding
        # trans_ct_fft_real = trans_ct_fft_real.permute(0, 3, 1, 2)  # (B,C,N,T)
        # residual_fft_real = trans_ct_fft_real
        # for decode_subnet in self.decode_subnets:
        #     residual_fft_real = residual_fft_real - decode_subnet(trans_ct_fft_real)
        # residual_fft_real = residual_fft_real.permute(0, 2, 3, 1)  # (B,N,T,C)
        #
        # # decode
        # revealed = self.RN(residual_fft_real, phase=None if not self.mag else hidden_phase)  # (B,secret_len)

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
        transform_ct_wav = watermark_ct_wav

        ## decode
        signal_fft = self.stft(transform_ct_wav)
        sign_fft_real = torch.view_as_real(signal_fft)
        watermark_fft_real = sign_fft_real  # obsismc: different from what the paper says
        _, message_restored_fft_real = self.enc_dec(sign_fft_real, watermark_fft_real, rev=True)
        message_restored_fft = torch.view_as_complex(message_restored_fft_real.contiguous())
        message_restored_expanded = self.istft(message_restored_fft)
        revealed = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)

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
        revealed = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        return revealed


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
