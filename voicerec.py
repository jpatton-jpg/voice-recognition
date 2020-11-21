# voicerec.py   #
# Joseph Patton #

import numpy as np
import scipy.io.wavfile
import pydub


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

