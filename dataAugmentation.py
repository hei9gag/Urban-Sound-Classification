import librosa
import librosa.display
import numpy as np
import pandas as pd
import os

timeAugmentaton = 0
pitchAugmentation = 1
csvFilePath = 'UrbanSound8K/metadata/UrbanSound8K.csv'
valid_data = None

def config():
    # Read Data
    data = pd.read_csv(csvFilePath)

    # Get data over 3 seconds long
    global valid_data
    valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= 3 ]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')

def dataAugmentation(augmentationType, rate):
    totalCount = 0
    progressThreashold = 100
    print('Total data:{}'.format(len(valid_data)))
    print('===========Import data begin===========')
    for row in valid_data.itertuples():
        if totalCount % progressThreashold == 0:
            print('Importing data row:{}'.format(totalCount))

        totalCount += 1

        wavForm, samplingRate = librosa.load('UrbanSound8K/audio/' + row.path, duration=2.97)
        ps = librosa.feature.melspectrogram(y=wavForm, sr=samplingRate)
        if ps.shape != (128, 128): continue

        if augmentationType == timeAugmentaton:
            changedWavForm = librosa.effects.time_stretch(wavForm, rate=rate)
            directory = 'augmented/fold' + str(row.fold) + '/speed_' + str(int(rate*100))
        else:
            changedWavForm = librosa.effects.pitch_shift(wavForm, samplingRate, n_steps=rate)
            directory = 'augmented/fold' + str(row.fold) + '/ps_' + str(int(rate))

        if not os.path.exists(directory):
            os.makedirs(directory)

        librosa.output.write_wav(directory + '/' + row.slice_file_name , changedWavForm, samplingRate)
    print('===========Import data finish===========')

if __name__ == '__main__':
    config()
    dataAugmentation(timeAugmentaton, 0.87)
    dataAugmentation(timeAugmentaton, 1.07)
    dataAugmentation(pitchAugmentation, -2)
    dataAugmentation(pitchAugmentation, 2)