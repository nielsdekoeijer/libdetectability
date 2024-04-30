# Detectability Perceptual Model in Python
A drop-in replacement for the MSE with a perceptual foundation.
Based on the Detectability model by van de Par et al. which can be found [here](https://link.springer.com/content/pdf/10.1155/ASP.2005.1292.pdf).
Includes a `Detectability` class to calculate the detectability through the `frame` and `frame_absolute` functions.
 - `frame(reference, test)`: Calculates the detectability of the error signal (`test` - `reference`) in presence of the `reference` signal.
 - `frame_absolute(reference, test)`: Calculates the detectability of the `test` signal in presence of the `reference` signal.

## Pytorch
I've also included a `DetectabilityLoss` class that allows one to use the `Detectability` as a pytorch loss function.
It currently assumes you always batch your inputs along the first dimension.
E.g. you use it as follows:
* `criterion = DetectabilityLoss()`
* `criterion(reference, test)`
* Note that the `DetectabilityLoss` does perceptual analysis on the **first** input argument!
  * Hence the order of arguments is very important: one should pass the `reference` as the first argument (no gradient, typically playing the role of a "label") and the changed `test` signal as the second argument (gradient, model output)!
 
## Disclaimer
This is my version of the detectability, thus subject to my interpretation of the original work and consequently may contain errors.
I am not an expert in modeling human perception, so I recommend double checking my work (and if mistakes are found, get in touch).
That being said this implementation has been used to great effect in a variety of signal processing project.

## Use
Upon creation of the detectability class, the user can specify a number of variables or leave them at the default setting:
 - `frame_size`: (2048) the length of the audio segments in samples.
 - `sampling_rate`: (48000) the sampling rate in Hz.
 - `taps`: (32) the number of gammatone filters used to cover the frequency range 0 Hz - sampling_rate / 2.
 - `dbspl`: (94.0):
 - `spl`: (1.0):
 - `relax_threshold`: (False) flag for determining whether to remove the threshold of hearing from the detectability calculation.
 - `normalize_gain`: (False) flag for determining whether to normalize the weighting vector associated with the detectability computation to a vector of unit-length.
 - `norm`: ("backward") option for determining in which direction it is desired for the underlying fft to perform the normalization.

## Experimental feature: segmented detectability
As an experimental feature, the detectability can now be used to evaluate the detectability long segments of audio which are divided into smaller segments through the use of the libsegmenter found [here](https://github.com/nielsdekoeijer/libsegmenter). The functionality is provided by the class `segmented_detectability` which takes the same input arguments as the regular `detectabilit` class. It will divide the reference and test signals into segments using a Hann window of the specified frame_size with 50% overlap. The computation is performed using the functions `calculate_segment_detectability_relative(reference, test)` and `calculate_segment_detectability_absolute(reference, test)`, which utilizes the `frame` and `frame_absolute` functions, respectively. The expected input dimensions are `[number_of_batch_elements, number_of_samples_in_each_element]` and the output dimensions are `[number_of_batch_elements, number_of_segments]`, i.e., the detectability is calculated for each segment independently.

## Contributors:
- Niels Evert Marinus de Koeijer
- Tudor-Razvan Tatar
- Martin Møller
