# Detectability Perceptual Model in Python
A drop-in replacement for the MSE with a perceptual foundation.
Based on the Detectability model by van de Par et al. which can be found [here](https://link.springer.com/content/pdf/10.1155/ASP.2005.1292.pdf).
Includes a `Detectability` class to calculate the detectability through the `frame` function.

I've also included a `DetectabilityLoss` class that allows one to use the `Detectability` as a pytorch loss function.
It currently assumes you always batch your inputs along the first dimension.
