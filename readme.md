
## Robotic Instrument Segmentation challenge

This repo contains my solution for the Robotic Instrument Segmentation Sub-Challenge which was part of the Endoscopic Vision Challenge held in conjunction with MICCAI 2017. The challenge webpage can be found [here](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)

![Fig 1. An illustration of the binary segmentation task segmentation.](demo.png)
Fig 1. An illustration of the binary segmentation task segmentation.

### Contents
1. [Explanation](#explanation)
2. [Requirements](#requirements)
3. [Demo](#demo)
4. [Contact](#contact)
5. [Acknowledgements](#acknowledgements)
6. [License](#license)


## Explanation 
In the contest, I implemented a "patch and stitch" CNN strategy followed by CRF post processing.

Essentially, a VGG-19 is trained on 51x51 patches from the input image. The patches containing any part of the medical instrument are labelled foreground and the rest background. Thus, the CNN acts like a foreground detector and is trained to learn a rough contour of the medical instrument.

A CRF is applied to the coarse output of the CNN. I use the CNN output as the unary potential. A binary potential is also applied. This basically 'cleans' up the image by identifying edges of the instrument which are characterised by large colour differences.

## Requirements
To run this code, you require the following softwares.
1. Anaconda for python 2.7 - Follow instructions [here](https://docs.continuum.io/anaconda/install/).
2. Pytorch - Get [here](http://pytorch.org/).
3. To get Pytorch to work on GPU, you need to have CUDA.

## Demo

A pre-trained foreground detector VGG-19 model can be downloaded from [here](https://drive.google.com/file/d/0B7oMEdOD2EhCQTljcldtWXNLaTg/view?usp=sharing). This model has been trained on  8 video sequences of 225 frames each, that were part of the challenge dataset.

To run the demo

1. Download and place the trained model in `Code/models`.
2. Follow the code within `Code/test_patch_cnn_and_crf.ipynb` to see the CNN and CRF results.

## Contact

For any assistance with the code or for reporting errors, please get in touch at rahulduggal2608 [at] gmail [dot] com.

## Acknowledgements
I would like to thank the authors of the following repos/blogs -
1. For implementations of standard CNN models - [this](https://github.com/kuangliu/pytorch-cifar).
2. For the CRF implementation - [this](http://www2.warwick.ac.uk/fac/sci/dcs/research/tia/software/sntoolbox/).
3. Good explanation of CNN+CRFs - [this](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/).

## License
This code is released under the MIT License (refer to the LICENSE file for details).
