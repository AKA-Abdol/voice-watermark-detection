import numpy as np
import matplotlib.pyplot as plt
import librosa

array, sampling_rate = librosa.load(librosa.ex("trumpet"))

D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure()#.set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()