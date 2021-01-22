import glob
import librosa
import os
import numpy as np
# data_from = "./data/LJSpeech/wavs"
# data_to = "./data/LJSpeech/mels"

data_from = "./data/LJSpeech/wavs_sample"
data_to = "./data/LJSpeech/mels_sample"

filelist = glob.glob(data_from + "/*")

if not os.path.exists(data_to):
    os.mkdir(data_to)

for i, wav_path in enumerate(filelist):
    name = wav_path.split('/')[-1].split('.')[0]
    y, sr = librosa.core.load(wav_path)
    mel = librosa.feature.melspectrogram(y, sr)
    np.save(os.path.join(data_to, name + ".npy"), mel )

print(f"sampling rate : {sr}")
print(f"converting done")

