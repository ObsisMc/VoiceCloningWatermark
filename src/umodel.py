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
import torch.nn.functional as F
from src.loader import AudioProcessor


def rgb_to_ycbcr(img):
    # Taken from https://www.w3.org/Graphics/JPEG/jfif3.pdf
    # img is mini-batch N x 3 x H x W of an RGB image

    output = torch.zeros(img.shape).to(img.device)

    output[:, 0, :, :] = 0.2990 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1114 * img[:, 2, :, :]
    output[:, 1, :, :] = -0.1687 * img[:, 0, :, :] - 0.3313 * img[:, 1, :, :] + 0.5000 * img[:, 2, :, :] + 128
    output[:, 2, :, :] = 0.5000 * img[:, 0, :, :] - 0.4187 * img[:, 1, :, :] - 0.0813 * img[:, 2, :, :] + 128

    return output


def ycbcr_to_rgb(img):
    # Taken from https://www.w3.org/Graphics/JPEG/jfif3.pdf
    # img is mini-batch N x 3 x H x W of a YCbCr image

    output = torch.zeros(img.shape).to(img.device)

    output[:, 0, :, :] = img[:, 0, :, :] + 1.40200 * (img[:, 2, :, :] - 128)
    output[:, 1, :, :] = img[:, 0, :, :] - 0.34414 * (img[:, 1, :, :] - 128) - 0.71414 * (img[:, 2, :, :] - 128)
    output[:, 2, :, :] = img[:, 0, :, :] + 1.77200 * (img[:, 1, :, :] - 128)

    return output


def pixel_unshuffle(img, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = img.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=img.device, dtype=img.dtype)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(img, kernel, stride=downscale_factor, groups=c)


def stft(self, data):
    window = torch.hann_window(self.n_fft).to(data.device)
    tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
    # [1, 501, 41, 2]
    return tmp


def istft(data, n_fft, hop_length):
    window = torch.hann_window(n_fft).to(data.device)
    return torch.istft(data, n_fft=n_fft, hop_length=hop_length, window=window,
                       return_complex=False)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, img):
        return pixel_unshuffle(img, self.downscale_factor)


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
        self.im_encoder_layers = nn.ModuleList([
            Down(1 + (not self.mag), 64),
            Down(64, 64 * 2)
        ])
        self.im_decoder_layers = nn.ModuleList([
            Up(64 * 2, 64),
            Up(64, 1 + (not self.mag))
        ])

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        return tmp

    def forward(self, seq):
        # TODO obsismc: the batch size must be 1
        seq = self.fc(seq)  # (B,L)
        seq_fft_img = self.stft(seq)
        seq_fft_real = torch.view_as_real(seq_fft_img) # (B,N,T,C)
        mag, phase = seq_fft_real[..., 0].unsqueeze(-1), seq_fft_real[..., 1].unsqueeze(-1)
        if self.mag:
            seq_fft_real = mag
        seq_fft_real = seq_fft_real.permute(0, 3, 1, 2)  # (B,C,N,T)

        seq_wavs_down = [seq_fft_real]
        # Encoder part of the UNet
        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            seq_wavs_down.append(enc_layer(seq_wavs_down[-1]))

        seq_wavs_up = [seq_wavs_down.pop()]

        # Decoder part of the UNet
        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            seq_wavs_up.append(dec_layer(seq_wavs_up[-1], seq_wavs_down[-1 - dec_layer_idx], None))

        return seq_wavs_up[-1].permute(0, 2, 3, 1), phase  # (B,N,T,C), (B,N,T,1)


class RevealNet(nn.Module):
    def __init__(self, mp_decoder=None, stft_small=True, embed='stretch', luma=False, secrete_len=32,
                 transform='fourier',
                 num_points=64000,
                 n_fft=1000,
                 hop_length=400,
                 mag=False):
        super(RevealNet, self).__init__()

        self.mp_decoder = mp_decoder
        self.pixel_unshuffle = PixelUnshuffle(2)
        self._stft_small = stft_small
        self.embed = embed
        self.luma = luma

        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag = mag

        self.im_encoder_layers = nn.ModuleList([
            Down(1 + (not self.mag), 64),
            Down(64, 64 * 2)
        ])
        self.im_decoder_layers = nn.ModuleList([
            Up(64 * 2, 64),
            Up(64, 1 + (not self.mag))
        ])

        self.fc = nn.Linear(self.num_points, secrete_len)

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def forward(self, ct, phase=None):
        ct = ct.permute(0, 3, 1, 2)  # (B,C,N,T)
        ct_down = [ct]

        # Encoder part of the UNet
        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            ct_down.append(enc_layer(ct_down[-1]))

        ct_up = [ct_down.pop(-1)]

        # Decoder part of the UNet
        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            ct_up.append(
                dec_layer(ct_up[-1],
                          ct_down[-1 - dec_layer_idx])
            )

        revealed = ct_up[-1]

        # obsismc: sequence
        revealed = revealed.permute(0, 2, 3, 1).contiguous()  # (B,N,T,C)
        if self.mag:
            spec_img = torch.view_as_complex(torch.cat([revealed, phase], dim=-1))
        else:
            spec_img = torch.view_as_complex(revealed)
        revealed = self.istft(spec_img)
        revealed = self.fc(revealed)

        return revealed  # (B,secret_len)


class StegoUNet(nn.Module):
    def __init__(self, transform='cosine', stft_small=True, ft_container='mag', mp_encoder='single',
                 mp_decoder='double', mp_join='mean', permutation=False, embed='stretch', luma='luma',
                 num_points=63600, n_fft=1022, hop_length=400, mag=False):

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

        if self.ft_container == 'magphase' and self.embed != 'stretch':
            raise Exception('Mag+phase does not work with embeddings other than stretch')
        if self.luma and self.embed == 'multichannel':
            raise Exception('Luma is not compatible with multichannel')

        if transform != 'fourier' or ft_container != 'magphase':
            self.mp_decoder = None  # For compatiblity with RevealNet

        # Sub-networks
        self.PHN = PrepHidingNet(self.transform, self.stft_small, self.embed, num_points=self.num_points,
                                 n_fft=self.n_fft, hop_length=self.hop_length, mag=self.mag)
        self.RN = RevealNet(self.mp_decoder, self.stft_small, self.embed, self.luma, num_points=self.num_points,
                            n_fft=self.n_fft, hop_length=self.hop_length, mag=self.mag)

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def forward(self, secret, cover, cover_phase=None):
        # cover_phase is not None if and only if using mag+phase
        # If using the phase only, 'cover' is actually the phase!
        # obsismc: image secret (B,C,256,256), cover (B,1024,512)
        # obsismc: sequence secret (B,32), cover (B,1,1024,512)
        assert not ((self.transform == 'fourier' and self.ft_container == 'magphase') and cover_phase is None)
        assert not ((self.transform == 'fourier' and self.ft_container != 'magphase') and cover_phase is not None)

        # Encode the image using PHN
        hidden_signal, hidden_phase = self.PHN(secret)  # (B,N,T,C)

        # Residual connection
        # Also keep a copy of the unpermuted containers to compute the loss
        cover_fft_img = self.stft(cover)
        cover_fft_real = torch.view_as_real(cover_fft_img)   # (B,N,T,2)
        mag, phase = cover_fft_real[..., 0].unsqueeze(-1), cover_fft_real[..., 1].unsqueeze(-1)
        if self.mag:
            cover_fft_real = mag
        container_fft = cover_fft_real + hidden_signal

        origin_ct_fft = container_fft  # (B,N,T,C)
        if self.mag:
            spec_img = torch.view_as_complex(torch.cat([container_fft, phase], dim=-1).contiguous())
        else:
            spec_img = torch.view_as_complex(container_fft.contiguous())
        origin_ct_wav = self.istft(spec_img)  # (B,L)

        revealed = self.RN(container_fft, phase=None if not self.mag else hidden_phase)  # (B,secret_len)
        return cover_fft_real, origin_ct_fft, origin_ct_wav, revealed
