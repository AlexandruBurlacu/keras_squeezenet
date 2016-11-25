# Keras Squeezenet
Keras implementation of Squeeze Net architecture described in arXiv : 1602.07360v3 (https://arxiv.org/pdf/1602.07360.pdf)

## TODOs:
* ~~add data augmentation~~ (Done)
* add bypasses
* experiment with Gaussian Noise
* ~~experiment with Batch Normalization~~ (Parametrized inside SqueezeNetBuilder and FireModule classes)
* ~~try more levels of Dense layers~~ (Added the possibility to inject a small subnet in the model)
