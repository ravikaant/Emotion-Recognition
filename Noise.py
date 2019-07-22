import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
folderpath='/home/shruti/Desktop/NITPatna/HopeIDK/3SecondsAboveData/SegmentedData/TestPaddedSegmented/'
fpath='/home/shruti/Desktop/NITPatna/Noises/SIGNAL003-20kHz.wav'
y1, sr1 =librosa.load(fpath)
y_1 = librosa.resample(y1, sr1, 16000)

for filename in os.listdir(folderpath):
        print(filename)
        filepath = folderpath + filename
        y, sr =librosa.load(filepath)
        y_s = librosa.resample(y, sr, 16000)
        sr = 16000
        n=y_1[:len(y_s)]
        n=n-np.mean(n)
        n=n/np.sqrt(np.var(n))
        vs=np.var(y_s)
        vn = vs/(np.power(10,(10/10)))  #SNR level, here 8dB
        n=n*np.sqrt(vn)
        nsp=y_s+n
        nsp=nsp/(max(abs(nsp)*1.01))
        #display original signal
        #plt.plot(y)
        #plt.show()
        #display noise signal
        #plt.plot(n)
        #plt.show()
        #display noisy signal
        #plt.plot(nsp)
        #plt.show()
        librosa.output.write_wav('/home/shruti/Desktop/NITPatna/HopeIDK/3SecondsAboveData/SegmentedData/NoiseDataset/NoiseTest_10DB_signal03/'+filename,nsp,sr)
