#!/usr/bin/env python

import sys
from preprocessing import prepare_single_file_data
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

def main(): 
    

    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' mode')
        print('available modes are single_file and test_info')
        return

    mode = sys.argv[1]
    model = load_model('choosen_model/model.hdf5')

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

    int_to_genres = dict(zip(range(10), genres))

    feat_train = np.load('preprocessed/array_data/features_train_data.npy')
    feature_scaler = StandardScaler()
    feature_scaler.fit(feat_train)


    if mode == 'single_file':
        file_path = raw_input('Choose song path: ')
       

        model_inputs = prepare_single_file_data(file_path, feature_scaler)
        predicted_class = model.predict(model_inputs)
    
        class_number_from_categorical = np.argmax(predicted_class[0])
        
        print('Model predicted ' + int_to_genres[class_number_from_categorical])     


    else:
        spec_test = np.load('preprocessed/array_data/spect_test_data.npy')
        feat_test = np.load('preprocessed/array_data/features_test_data.npy')
        classes_test = np.load('preprocessed/array_data/classes_test_data.npy')

        spec_test_normalized = spec_test.astype('float32') / 255.0
        feat_test_normalized = feature_scaler.transform(feat_test)

        classes_test_categorical = to_categorical(classes_test, num_classes=10)

        test_inputs = {'spectrogram_input' : spec_test_normalized, 'features_input' : feat_test_normalized}
        test_outputs = {'output' : classes_test_categorical}


        loss, acc = model.evaluate(test_inputs, test_outputs)
            
        print('Model evaluated.')
        print('Test loss: ' + str(loss))
        print('Test accuracy: ' + str(acc))
    
    
    

main()
