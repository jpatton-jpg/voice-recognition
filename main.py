# main.py       #
# Joseph Patton #

from voicerec import mp3_to_wav, parse_wav
from voicerec import spectral_centroid, get_fft
from voicerec import get_spectral_flux, get_rolloff
import numpy as np
import matplotlib.pyplot as plt
import librosa

mp3_file = 'obama.mp3'
wav_file = 'obama.wav'

# read audio file and convert to wav #
#mp3_to_wav(mp3_file,wav_file)

# get ch1 data and sampling rate from wav file #
Fs, audio_ch1 = parse_wav(wav_file)

bigN = audio_ch1.size  # Fs*t, total points in signal
N = 8192               # FFT size

# print some useful values #
print(f'Sampling Rate:           {Fs}')
print(f'Total Number of Samples: {bigN}')
print(f'FFT Size:                {N}')
print(f'# of Spectrums that will be generated: {bigN//N}')

from kaiser import wind  # get window function for fft as np array #

Y_old = np.zeros(N//2)
for i in range(bigN//N):
    # get 8192 samples #
    y = audio_ch1[N*i:N*(i+1)]

    # calc Mel-Frequency Cepstal Coefficients MFCC #
    mfcc = np.squeeze(librosa.feature.mfcc(y=y,sr=Fs,n_mfcc=13,hop_length=N+1))

    # calc zero-crossing rate #
    # value from 0 to 1       #
    zcr = np.count_nonzero(librosa.core.zero_crossings(y))/N

    # get real part of fft #
    f,Y = get_fft(y,wind,N,Fs)

    # calc spectral roll-off #
    # normalize from 0 to 1  #
    sro = get_rolloff(Y,N,f) / f[-1]

    # calc spectral centroid #
    # normalize from 0 to 1  #
    s_cent = spectral_centroid(f,Y) / f[-1]

    # calc spectral flux #
    flux = get_spectral_flux(Y_old,Y,Fs)
    Y_old = np.copy(Y)


    # plot spectrum #
    plt.style.use('ggplot')
    fig,ax = plt.subplots()
    plt.plot(f,Y,linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(sg, cmap='hot', aspect=(bigN//N)/(N//2))
#plt.ylabel('Frequency (Hz)')
#plt.xlabel('Window (8192 samples)')
#plt.show()
