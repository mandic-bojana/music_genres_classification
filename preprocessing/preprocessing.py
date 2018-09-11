#!/usr/bin/env python

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import scipy.ndimage
import librosa
from librosa.feature import melspectrogram
from librosa.display import specshow
from os import listdir



# computation of mel spectrogram
# mel scale provides spectral information in about the same way as human hearing)

def create_mel_spectrogram_file(rate, data, spect_img_file_path):
    
    freqs, times, spect = signal.spectrogram(data, rate, nfft=2048, nperseg=512, window=signal.windows.hann(512))
    mel_spect = melspectrogram(sr=rate, S=spect, n_mels=96, fmax=rate/2.0)
    
    plt.figure(figsize=(4, 4))
    fig, ax = plt.subplots(figsize=plt.figaspect(1))
    fig.subplots_adjust(0, 0, 1, 1)
    
    specshow(librosa.power_to_db(mel_spect, ref=np.max), fmax=rate/2.0)
    

    plt.savefig(spect_img_file_path)
    plt.close('all')



# dividing song into 3 parts
# if save_to_files == True, rate and file_name are required

def split_song_data(song_data, save_to_files=False, rate=None, file_name=None):
 
    length = len(song_data)
    
    fst_split_point = int(length / 3.0)
    snd_split_point = int(2.0/3 * length)
    
    part_1 = song_data[: fst_split_point]
    part_2 = song_data[fst_split_point : snd_split_point]
    part_3 = song_data[snd_split_point :]
    
    if save_to_files:
        wavfile.write(file_name + '_1' + '.wav', rate, part_1)
        wavfile.write(file_name + '_2' + '.wav', rate, part_2)
        wavfile.write(file_name + '_3' + '.wav', rate, part_3)
    
    return [part_1, part_2, part_3]



# FEATURE EXTRACTION -------------------------------------------------------



# amplitude mean and standard deviation

def amplitude_centroids(song_data):
    
    return np.mean(song_data), np.std(song_data)




# mean and standard deviation by frame of tempo (measured in beats per minute)

def tempo_features(song_data, rate):
    
    onset_env = librosa.onset.onset_strength(song_data.astype(np.float32), sr=rate)
    
    mean_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=rate, aggregate=np.mean).reshape((-1,))
    tempo_std = librosa.beat.tempo(onset_envelope=onset_env, sr=rate, aggregate=np.std).reshape((-1,))
    
    return mean_tempo[0], tempo_std[0]



# zero crossing rate (zcr) - number of points where the signal changes sign
# mean and standard deviation of zcr over frames

def zcr_features(song_data, rate, frame_length=2048, hop_length=512):
   
    zcr = librosa.feature.zero_crossing_rate(song_data.astype(np.float32),
                                             frame_length=frame_length, hop_length=hop_length)
    zcr = zcr.reshape((-1,))
    
    return np.mean(zcr), np.std(zcr)




# chroma features : total energy of the signal in each of 12 classes C, C#, D, D#, ... , A#, B
# mean and standard deviation over frames

def chroma_features(song_data, rate, n_fft=2048, hop_length=512):
    
    chroma = librosa.feature.chroma_stft(song_data.astype(np.float32), 
                                         sr=rate, n_fft=n_fft, hop_length=hop_length)
      
    
    return chroma.mean(axis=1), chroma.std(axis=1)



# each frame is divided into n_bands : spectral contrast is difference between min and max frequency
# in each band, and then mean and standard deviation over all frames is computed

def spectral_contrast_features(song_data, rate, n_fft=2048, hop_length=512, n_bands=5):
    
    contrast = librosa.feature.spectral_contrast(song_data.astype(np.float32), sr=rate, n_fft=n_fft,
                                                 hop_length=hop_length, n_bands=n_bands)
    
    
    return contrast.mean(axis=1), contrast.std(axis=1)





def main():

    for genre_name in listdir('../genres/'):
    
        for file_name in listdir('../genres/' + genre_name + '/'):
        
            rate, data = wavfile.read('../genres/' + genre_name + '/' + file_name)
            data_parts = split_song_data(data)
        
            spec_prefix = 'spectrograms/' + genre_name + '/' + file_name
        
            for i in range(len(data_parts)):
            
                create_mel_spectrogram_file(rate, data_parts[i], spec_prefix + '_' + str(i+1) + '.png')
            
                feature_vector = np.empty(0)
            
                amplitude_mean, amplitude_std = amplitude_centroids(data_parts[i])
            
                tempo_mean, tempo_std = tempo_features(data_parts[i], rate)
            
                zcr_mean, zcr_std = zcr_features(data_parts[i], rate)
            
                chroma_mean, chroma_std = chroma_features(data_parts[i], rate)
            
                contrast_mean, contrast_std = spectral_contrast_features(data_parts[i], rate)
            
            
            
                feature_vector = np.append(feature_vector, np.array([amplitude_mean, 
                                                                 amplitude_std, tempo_mean, tempo_std,
                                                                 zcr_mean, zcr_std]))
            
                feature_vector = np.append(feature_vector, chroma_mean)
                feature_vector = np.append(feature_vector, chroma_std)
                feature_vector = np.append(feature_vector, contrast_mean)
                feature_vector = np.append(feature_vector, contrast_std)

                

                feat_prefix = 'features/' + genre_name + '/' + file_name
        
                np.save(feat_prefix + '_' + str(i+1), feature_vector)

        
        print(genre_name + ' processed.')


main()
