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
from pystct import sdct_torch, isdct_torch
from loader import AudioProcessor
from torch_stft import STFT


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
                 n_fft=1000, hop_length=400):
        super(PrepHidingNet, self).__init__()
        self._transform = transform
        self._stft_small = stft_small
        self.embed = embed
        self._secrete_len = secrete_len
        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.fc = nn.Linear(self._secrete_len, self.num_points)  # TODO: may be optimized
        self.im_encoder_layers = nn.ModuleList([
            Down(2, 64),
            Down(64, 64 * 2)
        ])
        self.im_decoder_layers = nn.ModuleList([
            Up(64 * 2, 64),
            Up(64, 2)
        ])

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
        return tmp

    def forward(self, seq):
        # TODO obsismc: the batch size must be 1
        seq = self.fc(seq)  # (B,L)
        seq_fft = self.stft(seq)  # (B,N,T,C)
        seq_fft = seq_fft.permute(0, 3, 1, 2)  # (B,C,N,T)

        seq_wavs_down = [seq_fft]
        # print(f"seq_wavs_down[-1]: {seq_wavs_down[-1].shape}")
        # Encoder part of the UNet
        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            seq_wavs_down.append(enc_layer(seq_wavs_down[-1]))
            # print(f"seq_wavs_down[-1]: {seq_wavs_down[-1].shape}")

        seq_wavs_up = [seq_wavs_down.pop()]

        # Decoder part of the UNet
        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            # print(f"seq_wavs_up: {seq_wavs_up[-1].shape}, seq_wavs_down: {seq_wavs_down[-1 - dec_layer_idx].shape}")
            seq_wavs_up.append(dec_layer(seq_wavs_up[-1], seq_wavs_down[-1 - dec_layer_idx], None))

        return seq_wavs_up[-1].permute(0, 2, 3, 1)  # (B,N,T,C)


class RevealNet(nn.Module):
    def __init__(self, mp_decoder=None, stft_small=True, embed='stretch', luma=False, secrete_len=32,
                 transform='fourier',
                 num_points=64000,
                 n_fft=1000,
                 hop_length=400):
        super(RevealNet, self).__init__()

        self.mp_decoder = mp_decoder
        self.pixel_unshuffle = PixelUnshuffle(2)
        self._stft_small = stft_small
        self.embed = embed
        self.luma = luma

        self.num_points = num_points
        self.n_fft = n_fft
        self.hop_length = hop_length

        # If mp_decoder == unet or concatenating blocks, have RevealNet accept 2 channels as input instead of 1
        # if self.embed == 'blocks3' and not self._stft_small:
        #     self.im_encoder_layers = nn.ModuleList([
        #         Down(8, 64),
        #         Down(64, 64 * 2)
        #     ])
        #     self.im_decoder_layers = nn.ModuleList([
        #         Up(64 * 2, 64),
        #         Up(64, 1, opp_channels=8)
        #     ])
        # elif self.mp_decoder == 'unet' or self.embed == 'blocks3':
        #     self.im_encoder_layers = nn.ModuleList([
        #         Down(2, 64),
        #         Down(64, 64 * 2)
        #     ])
        #     self.im_decoder_layers = nn.ModuleList([
        #         Up(64 * 2, 64),
        #         Up(64, 1, opp_channels=2)
        #     ])
        # elif self.embed == 'multichannel':
        #     if self._stft_small:
        #         self.im_encoder_layers = nn.ModuleList([
        #             Down(8, 64),
        #             Down(64, 64 * 2)
        #         ])
        #         self.im_decoder_layers = nn.ModuleList([
        #             Up(64 * 2, 64),
        #             Up(64, 3, opp_channels=8)
        #         ])
        #     else:
        #         self.im_encoder_layers = nn.ModuleList([
        #             Down(32, 64),
        #             Down(64, 64 * 2)
        #         ])
        #         self.im_decoder_layers = nn.ModuleList([
        #             Up(64 * 2, 64),
        #             Up(64, 3, opp_channels=32)
        #         ])
        # else:
        self.im_encoder_layers = nn.ModuleList([
            Down(2, 64),
            Down(64, 64 * 2)
        ])
        self.im_decoder_layers = nn.ModuleList([
            Up(64 * 2, 64),
            Up(64, 2)
        ])

        self.fc = nn.Linear(self.num_points, secrete_len)

        # if self.embed == 'blocks2':
        #     if self._stft_small:
        #         self.decblocks = nn.Parameter(torch.rand(2))
        #     else:
        #         self.decblocks = nn.Parameter(torch.rand(8))
        # elif self.embed == 'blocks':
        #     self.decblocks = 1 / 2 * torch.ones(2) if self._stft_small else 1 / 8 * torch.ones(8)

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        return torch.istft(signal_wmd_fft, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                           return_complex=False)

    def forward(self, ct, ct_phase=None):
        # ct_phase is not None if and only if mp_decoder == unet
        # For other decoder types, ct is the only container
        # assert not (self.mp_decoder == 'unet' and ct_phase is None)
        # assert not (self.mp_decoder != 'unet' and ct_phase is not None)

        # Stretch the container to make it the same size as the image
        # if self.embed == 'stretch':
        #     ct = F.interpolate(ct, size=(256 * 2, 256 * 2))
        #     if self.mp_decoder == 'unet':
        #         ct_phase = F.interpolate(ct_phase, size=(256 * 2, 256 * 2))

        # if self.mp_decoder == 'unet':  # mp_decoder=None unless using magphase
        #     # Concatenate mag and phase containers to input to RevealNet
        #     ct_down = [torch.cat((ct, ct_phase), 1)]  # TODO obsismc: ct_phase is often None?
        # elif self.embed == 'blocks3':
        #     raise NotImplementedError
        #     if self._stft_small:
        #         # Undo split and concatenate in another dimension
        #         if self._stft_small:
        #             (rep1, rep2) = torch.split(ct, 512, 2)
        #         else:
        #             (rep1, rep2) = torch.split(ct, 1024, 2)
        #         ct_down = [torch.cat((rep1, rep2), 1)]
        #     else:
        #         # Split the 8 replicas and concatenate. 1x1x2048x1024 -> 1x8x512x512
        #         split1 = torch.split(ct, 512, 3)
        #         cat1 = torch.cat(split1, 1)
        #         split2 = torch.split(cat1, 512, 2)
        #         ct_down = [torch.cat(split2, 1)]
        # elif self.embed == 'multichannel':
        #     raise NotImplementedError
        #     # Small STFT: split the 8 replicas and concatenate. 1x1x1024x512 -> 1x8x256x256
        #     # Large STFT: split the 32 replicas and concatenate. 1x1x2048x1024 -> 1x32x256x256
        #     split1 = torch.split(ct, 256, 3)
        #     cat1 = torch.cat(split1, 1)
        #     split2 = torch.split(cat1, 256, 2)
        #     ct_down = [torch.cat(split2, 1)]
        # else:
        #     # Else there is only one container (can be anything)
        #     ct_down = [ct]
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

        # if self.embed == 'multichannel':
        #     raise NotImplementedError
        #     # The revealed image is the output of the U-Net
        #     revealed = ct_up[-1]
        # elif self.luma:
        #     raise NotImplementedError
        #     # Convert RGB to YUV, average lumas and back to RGB
        #     unshuffled = self.pixel_unshuffle(ct_up[-1])
        #     rgbs = torch.narrow(unshuffled, 1, 0, 3)
        #     luma = unshuffled[:, 3, :, :]
        #
        #     yuvs = rgb_to_ycbcr(rgbs)
        #     yuvs[:, 0, :, :] = 0.5 * yuvs[:, 0, :, :] + 0.5 * luma
        #
        #     revealed = ycbcr_to_rgb(yuvs)
        # else:
        #     # Pixel Unshuffle and delete 4th component
        #     # revealed = torch.narrow(self.pixel_unshuffle(ct_up[-1]), 1, 0, 3)
        #     revealed = ct_up[-1]
        revealed = ct_up[-1]

        # obsismc: sequence
        revealed = revealed.permute(0, 2, 3, 1)  # (B,N,T,C)
        revealed = self.istft(revealed)
        revealed = self.fc(revealed)

        # if self.embed == 'blocks' or self.embed == 'blocks2':
        #     raise NotImplementedError
        #     # Undo concatenation and recover a single image
        #     if self._stft_small:
        #         replicas = torch.split(revealed, 256, 2)
        #     else:
        #         replicas = torch.split(revealed, 256, 3)
        #         replicas = tuple([torch.split(replicas[i], 256, 2) for i in range(2)])
        #         replicas = replicas[0] + replicas[1]
        #     # Scale and add
        #     revealed = torch.sum(torch.stack([replicas[i] * self.decblocks[i] for i in range(len(self.decblocks))]),
        #                          dim=0)

        return revealed  # (B,secret_len)


class StegoUNet(nn.Module):
    def __init__(self, transform='cosine', stft_small=True, ft_container='mag', mp_encoder='single',
                 mp_decoder='double', mp_join='mean', permutation=False, embed='stretch', luma='luma',
                 num_points=63600, n_fft=1022, hop_length=400):

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

        if self.ft_container == 'magphase' and self.embed != 'stretch':
            raise Exception('Mag+phase does not work with embeddings other than stretch')
        if self.luma and self.embed == 'multichannel':
            raise Exception('Luma is not compatible with multichannel')

        if transform != 'fourier' or ft_container != 'magphase':
            self.mp_decoder = None  # For compatiblity with RevealNet

        # Sub-networks
        self.PHN = PrepHidingNet(self.transform, self.stft_small, self.embed, num_points=self.num_points,
                                 n_fft=self.n_fft, hop_length=self.hop_length)
        self.RN = RevealNet(self.mp_decoder, self.stft_small, self.embed, self.luma, num_points=self.num_points,
                            n_fft=self.n_fft, hop_length=self.hop_length)
        # if transform == 'fourier' and ft_container == 'magphase':
        #     # The previous one is for the magnitude. Create a second one for the phase
        #     if mp_encoder == 'double':
        #         self.PHN_phase = PrepHidingNet(self.transform, self.stft_small, self.embed)
        #     if mp_decoder == 'double':
        #         self.RN_phase = RevealNet(self.mp_decoder, self.stft_small, self.embed)
        #         if mp_join == '2D':
        #             self.mag_phase_join = nn.Conv2d(6, 3, 1)
        #         elif mp_join == '3D':
        #             self.mag_phase_join = nn.Conv3d(2, 1, 1)

        # if self.embed == 'blocks2' or self.embed == 'blocks3':
        #     if self.stft_small:
        #         self.encblocks = nn.Parameter(torch.rand(2))
        #     else:
        #         self.encblocks = nn.Parameter(torch.rand(8))

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
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

        # if self.embed != "multichannel":
        #     if self.luma:
        #         # Create a new channel with the luma values (R,G,B) -> (R,G,B,Y')
        #         lumas = rgb_to_ycbcr(secret)
        #         # Only keep the luma channel
        #         lumas = lumas[:,0,:,:].unsqueeze(1).to(secret.device)
        #         secret = torch.cat((secret,lumas),1)
        #     else:
        #         # Create a new channel with 0 (R,G,B) -> (R,G,B,0)
        #         zero = torch.zeros(secret.shape[0],1,256,256).type(torch.float32).to(secret.device)
        #         secret = torch.cat((secret,zero),1)

        # Encode the image using PHN
        hidden_signal = self.PHN(secret)
        # if self.transform == 'fourier' and self.ft_container == 'magphase' and self.mp_encoder == 'double':
        #     raise NotImplementedError
        #     hidden_signal_phase = self.PHN_phase(secret)

        # if self.embed == 'blocks' or self.embed == 'blocks2' or self.embed == 'blocks3':
        #     raise NotImplementedError
        #     if self.transform != 'fourier':
        #         raise Exception('\'blocks\' embedding is only implemented for STFT')
        #     # Replicate the hidden image as many times as required (only two for STFT)
        #     if self.embed == 'blocks':
        #         if self.stft_small:
        #             # Simply duplicate and concat vertically
        #             hidden_signal = torch.cat((hidden_signal, hidden_signal), 2)
        #         else:
        #             # Concat 8 copies of the image
        #             hidden_signal = torch.cat((hidden_signal, hidden_signal), 3)
        #             hidden_signal = torch.cat(tuple([hidden_signal for i in range(4)]), 2)
        #     else:
        #         # Else also scale with a learnable weight
        #         if self.stft_small:
        #             hidden_signal = torch.cat((hidden_signal * self.encblocks[0], hidden_signal * self.encblocks[1]), 2)
        #         else:
        #             hidden_signal1 = torch.cat(tuple([hidden_signal * self.encblocks[i] for i in range(4)]), 2)
        #             hidden_signal2 = torch.cat(tuple([hidden_signal * self.encblocks[i + 4] for i in range(4)]), 2)
        #             hidden_signal = torch.cat((hidden_signal1, hidden_signal2), 3)
        # elif self.embed == 'multichannel':
        #     raise NotImplementedError
        #     # Split the 8 channels and replicate. 1x8x256x256 -> 1x1x1024x512
        #     if self.stft_small:
        #         split1 = torch.split(hidden_signal, 2, dim=1)
        #     else:
        #         split1 = torch.split(hidden_signal, 4, dim=1)
        #     cat1 = torch.cat(split1, dim=2)
        #     split2 = torch.split(cat1, 1, dim=1)
        #     hidden_signal = torch.cat(split2, dim=3)

        # # Permute the encoded image if required
        # if self.permutation:
        #     raise NotImplementedError
        #     # Generate permutation index, which will be reused for the inverse
        #     perm_idx = torch.randperm(hidden_signal.nelement())
        #     # Permute the hidden signal
        #     hidden_signal = hidden_signal.view(-1)[perm_idx].view(hidden_signal.size())
        #     # Also permute the phase if necessary
        #     if self.transform == 'fourier' and self.ft_container == 'magphase' and self.mp_encoder == 'double':
        #         hidden_signal_phase = hidden_signal_phase.view(-1)[perm_idx].view(hidden_signal_phase.size())

        # Residual connection
        # Also keep a copy of the unpermuted containers to compute the loss
        cover_fft = self.stft(cover)  # (B,N,T,2)
        container_fft = cover_fft + hidden_signal
        # print(f"container_fft shape: {container_fft.shape}")
        origin_ct_fft = container_fft.clone()  # (B,N,T,C)
        origin_ct_wav = self.istft(origin_ct_fft)  # (B,L)
        # if self.transform == 'fourier' and self.ft_container == 'magphase':
        #     raise NotImplementedError
        #     if self.mp_encoder == 'double':
        #         container_phase = cover_phase + hidden_signal_phase
        #     else:
        #         container_phase = cover_phase + hidden_signal
        #     orig_container_phase = container_phase
        #
        # # Unpermute the encoded image if it was permuted
        # if self.permutation:
        #     raise NotImplementedError
        #     # Compute the inverse permutation
        #     inv_perm_idx = torch.argsort(perm_idx)
        #     # Permute the hidden signal with the inverse
        #     container = container.view(-1)[inv_perm_idx].view(container.size())
        #     # Also permute the phase if necessary
        #     if self.transform == 'fourier' and self.ft_container == 'magphase' and self.mp_encoder == 'double':
        #         container_phase = container_phase.view(-1)[inv_perm_idx].view(container_phase.size())

        # Reveal image
        # if self.transform == 'fourier' and self.ft_container == 'magphase':
        #     raise NotImplementedError
        #     if self.mp_decoder == 'unet':
        #         revealed = self.RN(container, container_phase)
        #     else:
        #         revealed = self.RN(container)
        #         revealed_phase = self.RN_phase(container_phase)
        #         if self.mp_join == 'mean':
        #             revealed = revealed.add(revealed_phase) * 0.5
        #         elif self.mp_join == '2D':
        #             join = torch.cat((revealed, revealed_phase), 1)
        #             revealed = self.mag_phase_join(join)
        #         elif self.mp_join == '3D':
        #             revealed = revealed.unsqueeze(1)
        #             revealed_phase = revealed_phase.unsqueeze(1)
        #             join = torch.cat((revealed, revealed_phase), 1)
        #             revealed = self.mag_phase_join(join).squeeze(1)
        #     return (orig_container, orig_container_phase), revealed
        # else:
        #     # If only using one container, reveal and return
        #     revealed = self.RN(container, ct_phase=phase).squeeze(0)
        #     return orig_container, revealed

        revealed = self.RN(container_fft)  # (B,secret_len)
        return cover_fft, origin_ct_fft, origin_ct_wav, revealed
