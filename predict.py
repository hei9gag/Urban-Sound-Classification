import numpy as np
import pandas as pd
import librosa
import sys
import glob
import os
from keras.models import load_model

def predict():
    df = pd.read_csv('class.csv')
    model = load_model('models/sound-classification.h5')
    wavFiles = glob.glob("predict/*.wav")

    for wavFile in wavFiles:
        y, sr = librosa.load(wavFile, duration=2.97)
        # exract features
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128):
            ps = _adjustedWavToAcceptableDuration(wavFile)
            if ps.shape != (128, 128): continue

        dataSet = []
        dataSet.append(ps)
        # reshape data to 128 x 128
        dataSet = np.array([data.reshape( (128, 128, 1) ) for data in dataSet])

        predictions = model.predict(dataSet)[0]
        print('============= Predict wav {} ============='.format(wavFile))
        for index, predict in enumerate(predictions):
            resultStr = '{0} {1:.2f}%'.format(df.iloc[index,1], predict * 100)
            print(resultStr)
        predictClass = model.predict_classes(dataSet)
        print('Result: {}'.format(df.iloc[predictClass[0],1]))
        print('============= Predict End =============')

# Try to duplicate the wav to make it at least 3 seconds long
def _adjustedWavToAcceptableDuration(wavFile):
    y, sr = _concatWavToDuration(wavFile, 2.97)
    librosa.output.write_wav('tmp.wav', y, sr)
    x, xr = librosa.load('tmp.wav', duration=2.97)
    os.remove('tmp.wav')
    return librosa.feature.melspectrogram(y=x, sr=xr)

def _concatWavToDuration(wavFile, duration):
    y, sr = librosa.load(wavFile, duration=duration)
    wavDuration = librosa.get_duration(y=y, sr=sr)
    if wavDuration >= duration:
        return (y, sr)
    concatTime = int(duration / wavDuration)
    concatedWav = [y]
    for _ in range(concatTime):
        concatedWav = np.append(concatedWav, y)
    return (concatedWav, sr)

if __name__ == '__main__':
    predict()