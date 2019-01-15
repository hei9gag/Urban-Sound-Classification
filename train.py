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
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Your data source for wav files
dataSourcePath = 'ToiletSoundSet/augmented'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 4

# model parameters for training
batchSize = 128
epochs = 12

# This function will import wav files by given data source path.
# And will extract wav file features using librosa.feature.melspectrogram.
# Class label will be extracted from the file name
# File name pattern: {WavFileName}-{ClassLabel}
# e.g. 0001-0 (0001 is the name for the wav and 0 is the class label)
# The program only interested in the class label and doesn't care the wav file name
def importData():
    dataSet = []
    totalCount = 0
    progressThreashold = 100
    dataSource = dataSourcePath
    for root, _, files in os.walk(dataSource):
        for file in files:
            fileName, fileExtension = os.path.splitext(file)
            if fileExtension != '.wav': continue
            if totalCount % progressThreashold == 0:
                print('Importing data count:{}'.format(totalCount))
            wavFilePath = os.path.join(root, file)
            y, sr = librosa.load(wavFilePath, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            if ps.shape != (128, 128): continue
            # extract the class label from the FileName
            label = fileName.split('-')[1]
            dataSet.append( (ps, label) )
            totalCount += 1
    global totalRecordCount
    totalRecordCount = totalCount
    return dataSet

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
def buildModel(dataset):
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
    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    model = Sequential()

    # Model Input
    input_shape=(128, 128, 1)

    # Using CNN to build model
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
    model.add(Dense(totalLabel))
    model.add(Activation('softmax'))

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test),
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
    buildModel(dataSet)