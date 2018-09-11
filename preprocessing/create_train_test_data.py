#!/usr/bin/env python

import numpy as np
from os import listdir
import cv2 as cv



def data_to_arrays(genres, genres_mapping):

    spectrogram_dir_name='spectrograms/'
    features_dir_name='features/'
    
    all_spectrograms_array = []
    all_features_array = []
    classes = []
    
    for genre in sorted(genres):
        
        spectrogram_features_pairs = zip(sorted(listdir(spectrogram_dir_name + genre + '/')),
                                               sorted(listdir(features_dir_name + genre + '/')))
        
        for spectrogram_img_file, features_file in spectrogram_features_pairs:
            
            spectrogram_img = cv.imread(spectrogram_dir_name + genre + '/' + spectrogram_img_file)
            spectrogram_img_resized = cv.resize(spectrogram_img, (216, 216))

            all_spectrograms_array.append(spectrogram_img_resized)
            
            
            features = np.load(features_dir_name + genre + '/' + features_file)
            
            all_features_array.append(features)
            
            
            song_class = genres_mapping[genre]
            
            classes.append(song_class)
            
    
    all_spectrograms_array = np.array(all_spectrograms_array)
    all_features_array = np.array(all_features_array)
    classes = np.array(classes)
    
    np.save('array_data/X_spectrograms', all_spectrograms_array)
    np.save('array_data/X_features', all_features_array)
    np.save('array_data/Y_classes', classes)
    
    


def main():

    
    genres = ['blues',
    'classical',
     'country',
     'disco',
     'hiphop',
     'jazz',
     'metal',
     'pop',
     'reggae',
     'rock']



    genres_to_int = dict(zip(genres, range(len(genres))))


    data_to_arrays(genres, genres_to_int)




main()
