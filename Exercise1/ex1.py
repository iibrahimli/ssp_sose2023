# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:03:40 2023

https://librosa.org/doc/latest/tutorial.html

@author: Natia_Mestvirishvili
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
    
def identify_voiced_region(signal):
    #normalize
    normalized_signal = signal / abs(max(s1[0]))

def identify_silent_regions(signal, sampling_rate):
    # short-time fourier transform
    n_fft = 512
    S = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft//2)
    # amplitudes to decibels
    decibels = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    #split the audio into non-silent regions by detecting audio higher than 60db
    silent_regions = librosa.effects.split(y=decibels, frame_length=sampling_rate, top_db=60)
    
    return silent_regions

# Source: https://www.researchgate.net/publication/269476477_Improving_Speech_Recognition_Rate_through_Analysis_Parameters
def my_windowing (signal, sampling_rate, frame_length, frame_shift):
    signal_length_in_ms = signal.shape[0]/(sampling_rate/1000)
    count_frames =  (signal_length_in_ms - frame_length) / frame_shift + 1
    return count_frames

# ***** 1 *****

#To preserve the native sampling rate of the file, use sr=None
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
s1_seg = s1[0][0:300]
s2_seg = s2[0][0:300]
plot_signal(s1_seg, sampling_rate_1, "Speech 1 silent segment")
plot_signal(s2_seg, sampling_rate_2, "Speech 2 silent segment")

# Voiced region: a region where the amplitudes are not close to zero, and also we can see
# a periodic trend
s1_seg = s1[0][25000:25300]
s2_seg = s2[0][24800:25100]
plot_signal(s1_seg, sampling_rate_1, "Speech 1 voiced segment")
plot_signal(s2_seg, sampling_rate_2, "Speech 2 voiced segment")

# Voiced region: a region where the amplitudes are not close to zero, and where we can see
# no periodic trend
s1_seg = s1[0][19600:19900]
s2_seg = s2[0][19200:19500]
plot_signal(s1_seg, sampling_rate_1, "Speech 1 unvoiced segment")
plot_signal(s2_seg, sampling_rate_2, "Speech 2 unvoiced segment")

#an audio file is an array(Nx1) of samples which represents the amplitude values 
#at a specific time instant (a sample)

#Sampling rate or sampling frequency defines the number of samples per second
#(or per other unit) taken from a continuous signal to make a discrete or digital signal.

#How can you detect the voiced and unvoiced regions of a speech signal?
#If the speech signal waveform looks periodic in nature, then it may be marked as voiced speech. 
#Otherwise, if the signal amplitude is low or negligible, then it can be marked as silence
#otherwise as unvoiced region.

# ***** 2 *****

#A signal is said to be stationary if its frequency or spectral contents are not changing 
# with respect to time. 

