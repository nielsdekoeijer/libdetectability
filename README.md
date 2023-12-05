# Detectability Perceptual Model in Python
A drop-in replacement for the MSE with a perceptual foundation.
Based on the Detectability model by van de Par et al. which can be found [here](https://link.springer.com/content/pdf/10.1155/ASP.2005.1292.pdf).
Includes a `Detectability` class to calculate the detectability through the `frame` function.

## Pytorch
I've also included a `DetectabilityLoss` class that allows one to use the `Detectability` as a pytorch loss function.
It currently assumes you always batch your inputs along the first dimension.
E.g. you use it as follows:
* `criterion = DetectabilityLoss()`
* `criterion(reference, test)`
* Note that the `DetectabilityLoss` does perceptual analysis on the **first** input argument!
  * Hence the order of arguments is very important: one should pass the `reference` as the first argument (no gradient, typically playing the role of a "label") and the changed `test` signal as the second argument (gradient, model output)!
## Contributors:
- Niels Evert Marinus de Koeijer
- Tudor-Razvan Tatar
