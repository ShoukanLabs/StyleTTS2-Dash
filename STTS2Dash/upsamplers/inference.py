import json
import os

import numpy as np
import torch

import torchaudio.functional as aF
from .model import APNet_BWE_Model



def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    log_amp = torch.log(torch.abs(stft_spec) + 1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack((torch.exp(log_amp) * torch.cos(pha),
                       torch.exp(log_amp) * torch.sin(pha)), dim=-1)

    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    amp = torch.exp(log_amp)
    com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return audio


def load_checkpoint(filepath, device="cpu"):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_model(config, path, device="cpu"):
    with open(config, "r", encoding="utf 8") as config:
        h = json.load(config)
    model = APNet_BWE_Model(h).to(device)

    state_dict = load_checkpoint(path, device)
    model.load_state_dict(state_dict['generator'])

    return model


def inference(old_sr, audio, config, model, device="cpu", chunk_seconds=15):
    model.to(device)

    with open(config, "r", encoding="utf-8") as config:
        h = json.load(config)

    model.eval()

    # Original sampling rate
    orig_sampling_rate = old_sr

    # Calculate chunk size in samples
    hr_chunk_size = int(chunk_seconds * h["hr_sampling_rate"])

    # Convert audio to tensor and add batch dimension
    audio = torch.tensor(np.float32(audio)).unsqueeze(0)  # Shape becomes (1, T)
    audio = audio.to(device)

    # Resample to high resolution
    audio_hr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=h["hr_sampling_rate"])

    # Prepare the output tensor
    output_chunks = []

    with torch.no_grad():
        # Process audio in chunks
        for i in range(0, audio_hr.size(1), hr_chunk_size):
            # Extract chunk from high-res audio
            chunk_hr = audio_hr[:, i:min(i + hr_chunk_size, audio_hr.size(1))]

            # Resample chunk to low resolution and back to high resolution
            chunk_lr = aF.resample(chunk_hr, orig_freq=h["hr_sampling_rate"], new_freq=h["lr_sampling_rate"])
            chunk_lr = aF.resample(chunk_lr, orig_freq=h["lr_sampling_rate"], new_freq=h["hr_sampling_rate"])

            # Ensure chunk_lr is trimmed to match chunk_hr's length
            chunk_lr = chunk_lr[:, :chunk_hr.size(1)]

            # Process the chunk
            amp_nb, pha_nb, com_nb = amp_pha_stft(chunk_lr, h["n_fft"], h["hop_size"], h["win_size"])

            amp_wb_g, pha_wb_g, com_wb_g = model(amp_nb, pha_nb)

            # Convert back to time domain
            chunk_hr_g = amp_pha_istft(amp_wb_g, pha_wb_g, h["n_fft"], h["hop_size"], h["win_size"])

            # Add the processed chunk to our list
            output_chunks.append(chunk_hr_g.squeeze().detach().cpu())

    # Concatenate all chunks
    final_output = torch.cat(output_chunks, dim=0)

    return final_output.numpy(), h["hr_sampling_rate"]