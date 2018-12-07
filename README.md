# Sound Classification using CNN
Sample code to show how to use the convolutional Neural Network (CNN) to build a model to classify sound

# Sound Dataset
UrbanSound8K
https://urbansounddataset.weebly.com/urbansound8k.html

Remark:
There are 2 set of sounds which are UrbanSound and UrbanSound8K. Please download UrbanSound8K

# Prerequisites
- Python 2.7
- Keras
- Librosa
- Numpy
- Pandas
- Virtualenv

# IDE
VsCode
https://code.visualstudio.com/download

# Setup Environment
I have only run the program on Mac OS
1. Use git or download the source code
2. Setup virtualenv in the source root folder by running virtualenv venv
3. Run **source venv/bin/activate**
4. Run **pip install -r requirements.txt** to install the library or pip install {library} one by one
5. Download SoundDataSet from UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html
6. Move UrbanSound8K to the source root folder

# How to train and predict using UrbanSound8K
**Train Model**
Please run python trainUrbanSound8K.py
The program will try to exract all wav features from UrbanSound8K folder, then build the model using CNN. At the end the program will export the models to folder in .h5 extension. You can use the model to predict .wav files.

**Predict**
Please rename the model you built to urban-sound.h5 and you can use it to predict .wav files.

First put the .wav files into predict folder. Then run
python predictUrbanSound8K.py

The result will show confidence level for each labels and the predict result
e.g.
```
air_conditioner 0.00%
car_horn 0.07%
children_playing 0.00%
dog_bark 0.00%
drilling 99.92%
engine_idling 0.00%
gun_shot 0.00%
jackhammer 0.00%
siren 0.00%
street_music 0.01%
Result: drilling
```
# Program Logic
The program logic is straightforward. It use librosa (the sound library) to convert the sound to spectrogram. Then it use the spectrogram as the sound feature.

Then it will reshape the data in 128 x 128 and use as an input format for the conventional Neural Network (CNN) to build the model to classify sound.

For each wav files, the naming convension must be in {wav-name}-{label}, because the program will use string split('-') to get the class label from the wav file. The class label will be used to teach the CNN the correct category for the sound file.

# Build Your Own Sound Classification
You can create your own dataset and import the data to the program like **UrbanSound8K**.

You can use **train.py** and **predict.py** as an reference. Also you need to update the **class.csv** for the sound classiifcation list.

