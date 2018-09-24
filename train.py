#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import keras.backend as K
from sklearn.metrics import f1_score
from keras import initializers
from model import create_model




def main():

    
    spec_train = np.load('preprocessed/array_data/spect_train_data.npy')
    feat_train = np.load('preprocessed/array_data/features_train_data.npy')
    classes_train = np.load('preprocessed/array_data/classes_train_data.npy')
    
    spec_val = np.load('preprocessed/array_data/spect_val_data.npy')
    feat_val = np.load('preprocessed/array_data/features_val_data.npy')
    classes_val = np.load('preprocessed/array_data/classes_val_data.npy')

    
    print('-- data divided into train, validation and split sections.')
    
    feature_scaler = StandardScaler()
    feature_scaler.fit(feat_train)
    
    feat_train_normalized = feature_scaler.transform(feat_train)
    feat_val_normalized = feature_scaler.transform(feat_val)
    print('-- features data normalized.')
    
    spec_train_normalized = spec_train.astype(np.float16) / 255.0
    spec_val_normalized = spec_val.astype(np.float16) / 255.0
    print('-- spectrogram data normalized.')
    
    
    classes_train_categorical = to_categorical(classes_train, num_classes=10)
    classes_val_categorical = to_categorical(classes_val, num_classes=10)
    print('-- class values converted to categorical.')
    
    model = create_model()
    print('-- model created.')
    
    plot_model(model, to_file='model.png')
    
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=[categorical_accuracy])
    print('-- model compiled.')
    
    train_inputs = {'spectrogram_input' : spec_train_normalized, 'features_input' : feat_train_normalized}
    train_outputs = {'output' : classes_train_categorical}
    
    
    val_inputs = {'spectrogram_input' : spec_val_normalized, 'features_input' : feat_val_normalized}
    val_outputs = {'output' : classes_val_categorical}

    
    print(model.summary())

    checkpoint = ModelCheckpoint(
        'callback_history/weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5', save_best_only=True, monitor='val_categorical_accuracy')


    epochs = 100
    decay_rate = 1.0 / epochs
    schedule = lambda epoch_index, lr: lr * (1. / (1. + decay_rate * epoch_index))
    lrScheduler = LearningRateScheduler(schedule, verbose=1)

    callbacks_list = [checkpoint, lrScheduler]
    
    model.fit(train_inputs, train_outputs, batch_size=128, epochs=epochs, 
                          validation_data=(val_inputs, val_outputs), callbacks=callbacks_list)
    
    model.save('model.hdf5')
    print('-- model saved.')



main()





