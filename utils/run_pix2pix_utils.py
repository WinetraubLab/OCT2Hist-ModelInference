# This util is a wrapper to the pix2pix algorithm and allows
# us to evaluate the model by providing an input image

import os
from google.colab import drive

# Temporary folder that we use to collect and organize files
# Base folder already contains pix2pix code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/9f8f61e5a375c2e01c5187d093ce9c2409f409b0
base_folder = "/content/OCT2Hist-UseModel/pytorch-CycleGAN-and-pix2pix"

# Folder containing the pre-trained model
model_folder = "/content/drive/Shareddrives/Yolab - Current Projects/_Datasets/2020-11-10 10x Model (Paper V2)"


def setup_network():
    """ Set up environment load the network parameters, run this code once """
    
    # Install dependencies
    !pip install -r {base_folder}/requirements.txt
    
    # Mount google drive (if not already mounted)
    drive.mount('/content/drive/')

    # Copy model parameters to the correct location
    !mkdir {base_folder}/checkpoints
    !mkdir {base_folder}/checkpoints/pix2pix/
    !cp "{model_folder}/latest_net_G.pth" {base_folder}/checkpoints/pix2pix/
    !cp "{model_folder}/latest_net_D.pth" {base_folder}/checkpoints/pix2pix/