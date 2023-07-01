# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:01:24 2023

"""
import numpy as np
import librosa
from scipy.signal import get_window
import matplotlib.pyplot as plt
import requests

plt.rcParams["figure.figsize"] = (14, 5)

# Prev exercise methods

def my_windowing(
    v_signal: np.ndarray,
    sampling_rate: int,
    frame_length: int,
    frame_shift: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_shift is None:
        frame_shift = frame_length

    frame_len_samples = np.floor(frame_length * sampling_rate / 1000).astype(int)
    frame_shift_samples = np.floor(frame_shift * sampling_rate / 1000).astype(int)
    m_frames = []
    frame_centers = []
    for i in range(0, len(v_signal) - frame_len_samples, frame_shift_samples):
        m_frames.append(v_signal[i : i + frame_len_samples])
        frame_centers.append((i + frame_len_samples // 2) * 1000 / sampling_rate)
    return np.array(m_frames), np.array(frame_centers)

def acf(frames):
    # convolve each frame with its time-reversed version
    acf_frames = [np.convolve(frame, frame[::-1]) for frame in frames]
    acf_frames = np.array(acf_frames)
    acf_frames = acf_frames[:, acf_frames.shape[1] // 2 :]
    return acf_frames

def estimate_fundamental_frequency(
    acf_frames: np.ndarray,
    sampling_freq: int,
    min_freq: float = 80,
    max_freq: float = 400,
) -> np.ndarray:
    """
    Estimate the fundamental frequency of each frame in acf_frames by searching
    for the maximum.
    """
    # convert bounds to acf indices
    min_lag = int(sampling_freq / max_freq)
    max_lag = int(sampling_freq / min_freq)
    f0s = np.argmax(acf_frames[:, min_lag:max_lag], axis=1) + min_lag
    return sampling_freq / f0s

def plot_signal(signal, sampling_freq, title):
    time = np.arange(signal.shape[0]) / sampling_freq
    plt.plot(time, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()
    
