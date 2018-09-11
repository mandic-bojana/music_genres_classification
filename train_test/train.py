#!/usr/bin/env python


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, concatenate, BatchNormalization
from keras.layers import MaxPool2D, Activation, Flatten, Dropout
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint



def split_data():
    
    spectrograms_data = np.load('../preprocessing/array_data/X_spectrograms.npy')
    features_data = np.load('../preprocessing/array_data/X_features.npy')
    song_classes = np.load('../preprocessing/array_data/Y_classes.npy')
    
    spec_train, spec_test, feat_train, feat_test, classes_train, classes_test = train_test_split(
                                                       spectrograms_data, 
                                                       features_data,
                                                       song_classes, test_size=0.2, random_state=3,
                                                       stratify=song_classes)
    
    return spec_train, spec_test, feat_train, feat_test, classes_train, classes_test



def create_model():
    
    spectrogram_input = Input(shape=(216, 216, 3), name='spectrogram_input')
    
    x1 = Conv2D(filters=6, kernel_size=(3, 3))(spectrogram_input)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2))(x1)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Flatten()(x1)
    
    x1 = Dense(units=120)(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Activation('relu')(x1)
    
    features_input = Input(shape=(42,), name='features_input')
    
    x2 = Dense(units=50)(features_input)
    x2 = Dropout(0.2)(x2)
    x2 = Activation('relu')(x2)
    
    x = concatenate([x1, x2])
    
    x = Dense(units=84)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(units=10)(x)
    
    output = Activation('softmax', name='output')(x)
    
    model = Model(inputs=[spectrogram_input, features_input], outputs=output)
    
    return model



def main():
    
    spec_train_val, spec_test, feat_train_val, feat_test, classes_train_val, classes_test = split_data()
    
    spec_train, spec_val, feat_train, feat_val, classes_train, classes_val = train_test_split(
                                                                                spec_train_val,
                                                                                feat_train_val,
                                                                                classes_train_val, 
                                                                                test_size=0.1, random_state=3,
                                                                                stratify=classes_train_val)
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
    
    
    checkpoint = ModelCheckpoint(
        'callback_history/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
    
    model.fit(train_inputs, train_outputs, batch_size=64, epochs=100, 
                          validation_data=(val_inputs, val_outputs), callbacks=[checkpoint])
    
    model.save('model.hdf5')
    print('-- model saved.')



main()





