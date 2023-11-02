from watermark_utils import wm_add_util, file_reader, wm_decode_util, my_parser, metric_util, path_util
from src.umodel import StegoUNet
import torch
import numpy as np
# from huggingface_hub import hf_hub_download


def load_model(path):
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state_dict = ckpt["state_dict"]
    model = StegoUNet(transform="ID", num_points=16000, n_fft=1000, hop_length=400,
                      mag=False, num_layers=4, watermark_len=32, shift_ratio=0)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def encode_watermark(model, signal, payload, pattern_bit_length=16, min_snr=20, max_snr=38, show_progress=False):
    device = next(model.parameters()).device

    pattern_bit = wm_add_util.fix_pattern[0:pattern_bit_length]

    watermark = np.concatenate([pattern_bit, payload])
    assert len(watermark) == 32
    signal_wmd, info = wm_add_util.add_watermark(watermark, signal, 16000, 0.1,
                                                 device, model, min_snr, max_snr,
                                                 show_progress=show_progress)
    info["snr"] = metric_util.signal_noise_ratio(signal, signal_wmd)
    return signal_wmd, info


def decode_watermark(model, signal, decode_batch_size=10, len_start_bit=16, show_progress=False):
    device = next(model.parameters()).device
    start_bit = wm_add_util.fix_pattern[0:len_start_bit]
    mean_result, info = wm_decode_util.extract_watermark_v3_batch(
        signal,
        start_bit,
        0.1,
        16000,
        model,
        device, decode_batch_size, show_progress=show_progress)

    if mean_result is None:
        return None, info

    payload = mean_result[len_start_bit:]
    return payload, info
