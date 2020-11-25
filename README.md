# voice-recognition

## TEMPORAL FEATURES USED ##
* Zero Crossing Rate

#### To add: #####
* Signal Energy
* Maximum Amplitude
* Minimum Energy

## SPECTRAL FEATURES USED ##
* Mel-Frequency Cepstral Coefficients (first 13 coefs)
* Spectral Centroid
* Spectral Flux
* Spectral Roll-off

#### To add: #####
* Fundamental Frequency
* Frequency Components
* Spectral Density
* Spectral Entropy
* Chroma Features
* Pitch

## RADIAL BASIS FUNCTION NN ##
* Activation function node number = input dimension number
* To find center of activation function, uses k-means clustering
* To find activation function widths, uses k-nearest neighbor
* To train weights, uses gradient descent


## NEEDED PACKAGES ##
* apt install sox
* apt install libsox-fmt-mp3

## Steps to convert from mp3 to wav and to remove silences ##
* sox ./filename.mp3 ./filename.wav
* sox filename.wav filename_ns.wav silence 1 0.1 1% -1 0.1 1%


