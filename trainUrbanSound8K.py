import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import TensorBoard
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import time
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# model parameters for training
batchSize = 128
epochs = 12

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
def importData():
    # Read Data
    data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

    # Get data over 3 seconds long
    valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= 3 ]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    print('Import data count:{}'.format(len(valid_data)))

    D = [] # Dataset
    totalCount = 0
    progressThreashold = 100
    print('===========Import data begin===========')
    for row in valid_data.itertuples():
        if totalCount % progressThreashold == 0:
            print('Importing data count:{}'.format(totalCount))
        y, sr = librosa.load('UrbanSound8K/audio/' + row.path, duration=2.97)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128): continue
        D.append( (ps, row.classID) )
        totalCount += 1
    print('===========Import data finish===========')
    global totalRecordCount
    totalRecordCount = totalCount
    return D

# Build model using Conventional Neural Network (CNN)
# Reference: https://www.youtube.com/watch?v=GNza2ncnMfA&t=502s
# https://www.youtube.com/watch?v=m8pOnJxOcqY
def trainData(dataset):
    print('TotalCount: {}'.format(totalRecordCount))
    trainDataEndIndex = int(totalRecordCount*0.8)
    random.shuffle(dataset)

    train = dataset[:trainDataEndIndex]
    test = dataset[trainDataEndIndex:]

    print('Total training data:{}'.format(len(train)))
    print('Total test data:{}'.format(len(test)))

    # Get the data (128, 128) and label from tuple
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Reshape for CNN input
    X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, 10))
    y_test = np.array(keras.utils.to_categorical(y_test, 10))

    model = Sequential()

    # Input
    input_shape=(128, 128, 1)

    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))

    # 31 x 62 x 24
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    # 27 x 58 x 48
    model.add(Conv2D(48, (5, 5), padding="valid"))

    # 6 x 29 x 48
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    # 2 x 25 x 48
    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    # Output
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/", histogram_freq=0,
                          write_graph=True, write_images=True)

    model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test),
        callbacks=[tensorboard]
    )

    score = model.evaluate(
        x=X_test,
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    dataSet = importData()
    trainData(dataSet)