# OCT to Histology - Using CycleGAN and Pix2Pix
We are trying to detect skin abnormalities without the need for invasive surgeries. With the help of Machine Learning, we are creating virtual histology directly from oct images. 

# Quick start
If you quickly want to get up and running with the Machine Learning part then:
- Do a git clone of the repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix <br>
there you'll find useful resources like datasets and everything you need to get started. 

# Prerequisites
We are running these models on Google Colab, and storing everying on a Google Drive.<br>
So, you may need enough storage on your Google cloud if you want to follow our approach exactly.<br><br>

Alternatively, if you don't want to run this on Google Colab, then you'll need:
- Linux or macOS 
- Python 3
- GPU

# How to set up
The most critical part of this project is to set up your data correctly.<br>
Since we are using paired (OCT->H&E) images, it's important that the model doesn't see the test images during training.


Make sure training images are where? and test images are ?

You need to combine the images and devide to training and testing, use these scripts. ....

# How to Train / Test
Use these colabs:
Before you run make sure that
1. Input images are placed in dataroot
2. Output images...
