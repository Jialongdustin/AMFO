from config import settings
import csv
import os.path as osp
import librosa

index1 = settings.librispeech_DIR
index2 = settings.librispeech_Index_DIR

Voice = []
with open(index2, encoding='utf-8') as f:
    reader = csv.reader(f)
    for i in reader:
        Voice.append(i[0])
a = librosa.load(osp.join(index1, Voice[0]), sr=22050)
print(a)





