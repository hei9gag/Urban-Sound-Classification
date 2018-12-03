import numpy as np
import pandas as pd
import librosa
import sys
from keras.models import load_model

model = load_model('models/sound-classification.h5')

# Predict Wav
y, sr = librosa.load('UrbanSound8K/audio/fold5/100032-3-0-0.wav', duration=2.97)
soundDuration = librosa.get_duration(y=y, sr=sr)

if soundDuration < 2.97:
    print('Sound durtaiton must be greater than 2.97 seconds')
    sys.exit()

ps = librosa.feature.melspectrogram(y=y, sr=sr)

if ps.shape != (128, 128):
    print('Is not in data shape (128, 128)')
dataSet = [] # Dataset
dataSet.append(ps)

dataSet = np.array([data.reshape( (128, 128, 1) ) for data in dataSet])

predictions = model.predict(dataSet)
predictClass = model.predict_classes(dataSet)
print('predictions:{} predictClass:{}'.format(predictions[0], predictClass[0]))
