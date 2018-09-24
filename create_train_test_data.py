#!/usr/bin/env python

import numpy as np
from os import listdir
import cv2 as cv



def data_to_arrays(genres, genres_mapping, spect_array_filename, features_array_filename, classes_array_filename, start_point, end_point):

    spectrogram_dir_name='preprocessed/spectrograms/'
    features_dir_name='preprocessed/features/'
    
    all_spectrograms_array = []
    all_features_array = []
    classes = []
    
    for genre in sorted(genres):

        spectrogram_features_pairs = zip(sorted(listdir(spectrogram_dir_name + genre + '/')),
                                               sorted(listdir(features_dir_name + genre + '/')))
        
        for spectrogram_img_file, features_file in spectrogram_features_pairs[start_point : end_point]:

            print(spectrogram_img_file, features_file)
            
            spectrogram_img = cv.imread(spectrogram_dir_name + genre + '/' + spectrogram_img_file)
            spectrogram_img_resized = cv.resize(spectrogram_img, (128, 128))
            spectrogram_img_gray = cv.cvtColor(spectrogram_img_resized, cv.COLOR_BGR2GRAY)

            all_spectrograms_array.append(spectrogram_img_gray)
            
            
            features = np.load(features_dir_name + genre + '/' + features_file)
            
            all_features_array.append(features)
            
            
            song_class = genres_mapping[genre]
            
            classes.append(song_class)
            
    
    all_spectrograms_array = np.array(all_spectrograms_array).reshape(-1, 128, 128, 1)
    all_features_array = np.array(all_features_array)
    classes = np.array(classes)
    
    np.save('preprocessed/array_data/' + spect_array_filename, all_spectrograms_array)
    np.save('preprocessed/array_data/' + features_array_filename, all_features_array)
    np.save('preprocessed/array_data/' + classes_array_filename, classes)
    
    


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

     
    data_to_arrays(genres, genres_to_int, 'spect_test_data', 'features_test_data', 'classes_test_data', start_point=0, end_point=20)    
    data_to_arrays(genres, genres_to_int, 'spect_val_data', 'features_val_data', 'classes_val_data', start_point=20, end_point=30)       
    data_to_arrays(genres, genres_to_int, 'spect_train_data', 'features_train_data', 'classes_train_data', start_point=30, end_point=300)
    

 

main()
