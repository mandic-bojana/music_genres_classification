from keras.models import Model
from keras.layers import Input, Dense, Conv2D, concatenate, BatchNormalization
from keras.layers import MaxPool2D, Activation, Flatten, Dropout


def create_model():
    
    spectrogram_input = Input(shape=(128, 128, 1), name='spectrogram_input')
    
    x1 = Conv2D(filters=16, kernel_size=(2, 2))(spectrogram_input)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
      
    x1 = Conv2D(filters=32, kernel_size=(2, 2))(x1)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)    

    x1 = Conv2D(filters=64, kernel_size=(2, 2))(x1)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Flatten()(x1)
    
    x1 = Dense(units=520)(x1)
    x1 = Dropout(0.4)(x1)
    x1 = Activation('relu')(x1)
    
    features_input = Input(shape=(42,), name='features_input')
    
    x2 = Dense(units=50)(features_input)
    x2 = Dropout(0.5)(x2)
    x2 = Activation('relu')(x2)
    
    x = concatenate([x1, x2])
    
    x = Dense(units=84)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(units=10)(x)
    
    output = Activation('softmax', name='output')(x)
    
    model = Model(inputs=[spectrogram_input, features_input], outputs=output)
    
    return model
