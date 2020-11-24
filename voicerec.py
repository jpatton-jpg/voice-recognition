# voicerec.py   #
# Joseph Patton #

import numpy as np
import scipy.io.wavfile
import pydub
import matplotlib.pyplot as plt
import librosa


def get_spectral_flux(X_old, X_cur, f_s):
    ''' computes the spectral flux from the magnitude spectrum
    https://www.audiocontentanalysis.org/code/audio-features/spectral-flux-2/
    '''
    # difference spectrum (set first diff to zero)
    X = np.c_[X_old, X_cur]
    afDeltaX = np.diff(X, 1, axis=1)
    # calculate flux #
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
    return vsf


def mp3_to_wav(in_filename,out_filename):
    ''' convert an mp3 file to wav '''
    # read mp3 file #
    mp3 = pydub.AudioSegment.from_mp3(in_filename)
    # convert to wav #
    mp3.export(out_filename, format='wav')
    return


def parse_wav(filename):
    ''' read wav file and find sample rate. return ch 1 '''
    Fs, audio_data = scipy.io.wavfile.read(filename)
    # take audio channel 1 #
    audio_ch1 = np.array([x[0] for x in audio_data])
    # filter out breaks in between words #
    audio_ch1 = audio_ch1[audio_ch1 > 40]
    # normalize #
    audio_ch1 = audio_ch1 / np.amax(audio_ch1)
    # center at 0 #
    audio_ch1 = audio_ch1 - np.average(audio_ch1)
    return Fs, audio_ch1


def spectral_centroid(f,real_part):
    ''' calculate spectral centroid '''
    return np.sum(f*real_part)/np.sum(real_part)


def get_fft(y,wind,N,Fs):
    ''' calculate N point FFT '''
    # window the data #
    y = y * wind
    # take fft #
    Y = np.fft.fft(y)[0:N//2]/N
    # single-sided spectrum only #
    Y[1:] = 2 * Y[1:]
    # generate x-axis #
    f = Fs * np.arange(N/2) / N;
    # take only real part #
    return f,np.abs(Y)


def get_rolloff(Y,N,f,rolloff=.85):
    ''' find the point F in the spectrum where rolloff percent
    of the energy in the spectrum is contained at or below F '''
    Y_sq = Y*Y
    csum = np.cumsum(Y_sq)
    cdf = csum / np.amax(csum)
    ind = (np.abs(cdf-rolloff)).argmin()
    return f[ind]


def get_data(mp3_file,isthisobama):
    # read audio file and convert to wav #
    #mp3_to_wav(mp3_file,mp3_file+'.wav')

    # get ch1 data and sampling rate from wav file #
    print('Parsing audio file...')
    Fs, audio_ch1 = parse_wav(mp3_file+'.wav')

    bigN = audio_ch1.size  # Fs*t, total points in signal
    N = 8192               # FFT size

    # print some useful values #
    #print('Audio file information: ')
    print(f'Audio Sampling Rate:     {Fs}')
    print(f'Total Number of Samples: {bigN}')
    #print(f'FFT Size:                {N}')
    #print(f'# of Spectrums that will be generated: {bigN//N}')

    # get window function for fft as np array #
    from kaiser import wind

    # put data in this array #
    output = np.zeros((bigN//N,5))

    # hold old spectrum to calculate flux #
    Y_old = np.zeros(N//2)

    print('Characterizing audio...')
    for i in range(bigN//N):
        # get 8192 samples #
        y = audio_ch1[N*i:N*(i+1)]

        # calc Mel-Frequency Cepstal Coefficients MFCC #
        mfcc = np.squeeze(librosa.feature.mfcc(y=y,sr=Fs,n_mfcc=13,
            hop_length=N+1))

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

        #output[i] = [*mfcc,zcr,sro,s_cent,flux,isthisobama]
        output[i] = [zcr,sro,s_cent,flux,isthisobama]

        # plot spectrum #
        #plt.style.use('ggplot')
        #fig,ax = plt.subplots()
        #plt.plot(f,Y,linewidth=1)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #plt.ylabel('Amplitude')
        #plt.xlabel('Frequency [Hz]')
        #plt.show()
    return output

