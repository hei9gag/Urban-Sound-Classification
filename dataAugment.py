import librosa
import numpy as np
import pandas as pd
import os
import shutil

CONCAT_WAV = 0
SPLIT_WAV = 1
STRETCH_WAV = 2
SHIFTED_WAV = 3

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

def _stretchWav(wavFile, duration, rate):
    y, sr = librosa.load(wavFile, duration=duration)
    stretchedWav = librosa.effects.time_stretch(y, rate=rate)
    return (stretchedWav, sr)

def _shiftWav(wavFile, duration, steps):
    y, sr = librosa.load(wavFile, duration=duration)
    shiftedWav = librosa.effects.pitch_shift(y, sr, n_steps=steps)
    return (shiftedWav, sr)

def _augmentData(source, out, augmentType, augmentValue, label, duration):
    counter = 1
    _createDirectoryIfNotExist(out)
    for root, _, files in os.walk(source):
        for file in files:
            _, fileExtension = os.path.splitext(file)
            if fileExtension != '.wav': continue
            wavFilePath = os.path.join(root, file)
            if augmentType == CONCAT_WAV:
                wav, sr = _concatWavToDuration(wavFilePath, augmentValue)
                fileName = str(counter).zfill(4) + '.wav'
            elif augmentType == SHIFTED_WAV:
                wav, sr = _shiftWav(wavFilePath, duration, augmentValue)

            if augmentType != CONCAT_WAV:
                fileName = str(counter).zfill(4) + '-' + str(label) + '.wav'
            librosa.output.write_wav(out + '/' + fileName, wav, sr)
            counter += 1

# Split the wav into duration
def _splitWav(direcoty, outDir, duration, label, maxSplit):
    counter = 1
    _createDirectoryIfNotExist(outDir)
    for root, _, files in os.walk(direcoty):
        for file in files:
            _, fileExtension = os.path.splitext(file)
            if fileExtension != '.wav': continue
            wavFilePath = os.path.join(root, file)

            # load wav
            y, sr = librosa.load(wavFilePath)
            wavDuration = librosa.get_duration(y=y, sr=sr)

            # calculate num of split
            splitTimes = min(maxSplit, int(wavDuration / duration))
            offSet = 0

            # split the wav into duration
            for _ in range(splitTimes):
                wav, rate = librosa.load(wavFilePath, offset=offSet, duration=duration)
                fileName = str(counter).zfill(4) + '-' + str(label) + '.wav'
                librosa.output.write_wav(outDir + '/' + fileName, wav, rate)
                offSet += duration
                counter += 1

def _createDirectoryIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    category = 'unknown'
    # 0: toilet_flush
    # 1: shower
    # 2: unknown
    # 3: door

    label = 2
    maxSoundDuration = 8
    maxSplit = 1
    soundDuration = 4

    inDir = 'ToiletSoundSet/' + category
    outDir = 'ToiletSoundSet/augmented/tmp'
    augmentedDirPrefix = 'ToiletSoundSet/augmented'
    sourceDir = augmentedDirPrefix + '/source/' + category

    # Convert the wav to at least 7 seconds
    print('======= concat wav begin =======')
    _augmentData(inDir, outDir, CONCAT_WAV, maxSoundDuration, label, soundDuration)

    # Split the wav file into 3 seconds wav
    print('======= split wav begin =======')
    _splitWav(outDir, sourceDir, soundDuration, label, maxSplit)

    shiftRates = [-1, 1, -2, 2, -2.5, 2.5, 3.5, -3.5]
    for rate in shiftRates:
        print('======= split rate:{} begin ======='.format(rate))
        augmentedOutputDir = augmentedDirPrefix + '/shifted/'+category+'/ps_' + str(rate)
        _augmentData(sourceDir, augmentedOutputDir, SHIFTED_WAV, rate, label, soundDuration)

    shutil.rmtree(outDir, ignore_errors=True)