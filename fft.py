# fft.py        #
# Joseph Patton #

import numpy as np
import scipy.io.wavfile
import pydub


def FeatureSpectralFlux(X, f_s):
    '''
    https://www.audiocontentanalysis.org/code/audio-features/spectral-flux-2/

    computes the spectral flux from the magnitude spectrum
      Args:
        X: spectrogram (dimension FFTLength X Observations)
        f_s: sample rate of audio data
      Returns:
        vsf spectral flux '''
    isSpectrum = X.ndim == 1
    if isSpectrum:
        X = np.expand_dims(X, axis=1)
    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    afDeltaX = np.diff(X, 1, axis=1)
    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
    return np.squeeze(vsf) if isSpectrum else vsf


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
    return Fs, audio_ch1


def spectral_centroid(f,real_part):
    ''' calculate spectral centroid '''
    return np.sum(f*real_part)/np.sum(real_part)


def get_fft(y,wind,N):
    ''' calculate N point FFT '''
    # window the data #
    y = y * wind
    # take fft #
    Y = np.fft.fft(y)[0:N//2]/N
    # single-sided spectrum only #
    Y[1:] = 2 * Y[1:]
    # take only real part #
    return np.abs(Y)


