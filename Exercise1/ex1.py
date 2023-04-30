# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:03:40 2023

"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_signal(signal, sampling_freq, title):
    plt.figure()
    librosa.display.waveshow(signal, sr=sampling_freq)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

def plot_signal_no_time(signal, sampling_freq, title):
    time = np.arange(signal.shape[0]) 
    plt.plot(time, signal)
    plt.xlabel("sample")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

# Source: https://www.researchgate.net/publication/269476477_Improving_Speech_Recognition_Rate_through_Analysis_Parameters
def my_windowing (signal, sampling_rate, frame_length, frame_shift):
    # Calculate signal length in milliseconds
    signal_length_in_ms = signal.shape[0]/(sampling_rate/1000)
    # Calculate number of frames from given input
    count_frames = int((signal_length_in_ms - frame_length) / frame_shift + 1)
    # convert frame shift and frame length from ms to number of samples
    frame_shift_in_samples = int((frame_shift / 1000) * sampling_rate)
    frame_length_in_samples = int((frame_length / 1000) * sampling_rate)
    # Divide signal into overlapping frames
    frames = librosa.util.frame(signal, frame_length=frame_length_in_samples, hop_length=frame_shift_in_samples, axis=0)
    return frames, count_frames

def amplitudes_to_dbs(signal):
    n_fft = 512
    S = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft//2)
    # amplitudes to decibels
    decibels = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return decibels

# ***** 1 *****

# To preserve the native sampling rate of the file, use sr=None
s1 = librosa.core.load("speech1.wav", sr=None)
s2 = librosa.core.load("speech2.wav", sr=None)
print("Speech 1: ", s1)
print("Speech 2: ", s2)

sampling_rate_1 = librosa.get_samplerate("speech1.wav")
sampling_rate_2 = librosa.get_samplerate("speech2.wav")
print("Sampling rate 1: ", sampling_rate_1)
print("Sampling rate 2: ", sampling_rate_2)

# s1[0] takes amplitudes from (Nx1) array
plot_signal(s1[0], sampling_rate_1, "Speech 1")
plot_signal(s2[0], sampling_rate_2, "Speech 2")

# To analyze voiced, unvoiced and silent regions, we can plot the waveform without time domain
plot_signal_no_time(s1[0], sampling_rate_1, "Speech 1")
plot_signal_no_time(s2[0], sampling_rate_1, "Speech 2")

# From there we can zoom in to regions of silence
s1_seg_silent = s1[0][0:512]
s2_seg_silent = s2[0][0:512]
plot_signal(s1_seg_silent, sampling_rate_1, "Speech 1 silent segment")
plot_signal(s2_seg_silent, sampling_rate_2, "Speech 2 silent segment")

# Voiced region: a region where the amplitudes are not close to zero, and also we can see
# a periodic trend
s1_seg_voiced = s1[0][24488:25512]
s2_seg_voiced = s2[0][24800:25824]
plot_signal(s1_seg_voiced, sampling_rate_1, "Speech 1 voiced segment")
plot_signal(s2_seg_voiced, sampling_rate_2, "Speech 2 voiced segment")

# Eye estimated fundamental frequency for speech 1 = 1/0.005 = 200
# Eye estimated fundamental frequency for speech 2 = 1/0.01 = 100

# Unvoiced region: a region where the amplitudes are not close to zero, and where we can see
# no periodic trend
s1_seg_unvoiced = s1[0][19600:20112]
s2_seg_unvoiced = s2[0][19700:19956]
plot_signal(s1_seg_unvoiced, sampling_rate_1, "Speech 1 unvoiced segment")
plot_signal(s2_seg_unvoiced, sampling_rate_2, "Speech 2 unvoiced segment")

# An audio file is an array(Nx1) of samples which represents the amplitude values 
# at a specific time instant (a sample)

# Sampling rate or sampling frequency defines the number of samples per second
# (or per other unit) taken from a continuous signal to make a discrete or digital signal.

# How can you detect the voiced and unvoiced regions of a speech signal?
# If the speech signal waveform looks periodic in nature, then it may be marked as voiced speech. 
# Otherwise, if the signal amplitude is low or negligible, then it can be marked as silence
# otherwise as unvoiced region.

# ***** 2 *****

# A signal is said to be stationary if its frequency or spectral contents are not changing 
# with respect to time. 

# The frame shift is the time difference between the start points of successive frames, 
# and the frame length is the time duration of each frame.

frames_s1, count_frames_s1 = my_windowing(s1_seg_voiced, sampling_rate_1, 32, 16)
frames_s2, count_frames_s2 = my_windowing(s2_seg_voiced, sampling_rate_1, 32, 16)

# ***** 3 *****

acf = np.zeros(1023,)
for frame in frames_s1:
    acf_frame = np.convolve(frames_s1[0], frames_s1[0][::-1]) / 512
    acf = acf + acf_frame

acf = acf[int(len(acf)/2):]
acf = acf[80:400]
print(np.argmax(acf))

plot_signal_no_time(acf, sampling_rate_1, "ACF signal 1")

acf = np.zeros(1023,)
for frame in frames_s1:
    acf_frame = np.convolve(frames_s2[0], frames_s2[0][::-1]) / 512
    acf = acf + acf_frame

acf = acf[int(len(acf)/2):]
acf = acf[80:400]
print(np.argmax(acf))

plot_signal_no_time(acf, sampling_rate_1, "ACF signal 2")

