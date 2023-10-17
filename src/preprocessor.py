import os
import torch
import torchaudio
import datasets
import tqdm
import whisper
import time


def save_sample(save_path, audio, sr, text):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torchaudio.save(os.path.join(save_path, "speech.wav"), audio, sr)
    with open(os.path.join(save_path, "text.txt"), "w") as f:
        f.write(text)


def build_training_dataset(num_points, output_dir, base_n, demo=True):
    assert os.path.exists(output_dir)

    if demo:
        dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_demo", "clean")
        datas = dataset["validation"]
    else:
        dataset = datasets.load_dataset("librispeech_asr")
        datas = dataset["train.clean.100"]
    num = len(datas)

    bar = tqdm.tqdm(total=num)
    n = base_n
    whisper_m = whisper.load_model("base")
    for data in datas:
        _, audio, sr = data["audio"].values()
        audio = torch.tensor(audio, dtype=torch.float32)

        len_audio = audio.size(-1)
        num_slices = len_audio // num_points
        remain_slice = len_audio - num_slices * num_points
        for i in range(num_slices):
            wav_slice = audio[i * num_points: (i + 1) * num_points]
            text_slice = whisper_m.transcribe(wav_slice)["text"]
            save_sample(os.path.join(output_dir, str(n)), wav_slice[None, ...], sr, text_slice)
            n += 1

        if remain_slice / num_points > 0.5:
            wav_slice = audio[-remain_slice:]
            text_slice = whisper_m.transcribe(wav_slice)["text"]
            save_sample(os.path.join(output_dir, str(n)), wav_slice[None, ...], sr, text_slice)
            n += 1

        bar.update()

    print(f"total number: {n}")


if __name__ == "__main__":
    num_points = 48000
    save_root = "YOUR_PATH"
    build_training_dataset(num_points, save_root, base_n=0, demo=False)
