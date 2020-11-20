# main.py       #
# Joseph Patton #

from fft import mp3_to_wav,parse_wav
from fft import spectral_centroid,get_fft
from fft import FeatureSpectralFlux
import numpy as np
import matplotlib.pyplot as plt

# read audio file and convert to wav #
mp3_file = 'obama_speeches.mp3'
wav_file = 'obama.wav'
#mp3_to_wav(mp3_file,wav_file)

# get ch1 data and sampling rate from wav file #
Fs, audio_ch1 = parse_wav(wav_file)

bigN = audio_ch1.size  # Fs*t, total points in signal
N = 8192               #

# print some useful values #
print(f'Sampling Rate: {Fs}')
print(f'Total Number of Samples: {bigN}')
print(f'FFT Size: {N}')
print(f'# of Spectrums that will be generated: {bigN//N}')

from kaiser import wind  # window function as list #
wind = np.array(wind)    # change to np.array      #

#sg = np.zeros((N//2,bigN//N))

for i in range(bigN//N):
    # get 8192 samples #
    y = audio_ch1[N*i:N*(i+1)]

    # get real part of fft #
    real_part = get_fft(y,wind,N)

    # generate x-axis #
    f = Fs * np.arange(N/2) / N;

    # calculate spectral centroid #
    s_cent = spectral_centroid(f,real_part)

    # calculate spectral flux #
    flux = FeatureSpectralFlux(real_part,Fs)
    print(flux)

    #sg[:,i] = np.log(np.transpose(real_part))

    # plot spectrum #
    plt.style.use('ggplot')
    fig,ax = plt.subplots()
    plt.plot(f,real_part,linewidth=1)
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
