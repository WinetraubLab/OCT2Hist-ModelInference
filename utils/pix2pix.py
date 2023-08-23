# This util is a wrapper to the pix2pix algorithm and allows
# us to evaluate the model by providing an input image

import cv2
import numpy as np
import os
import subprocess
from google.colab import drive

# Temporary folder that we use to collect and organize files
# Base folder already contains pix2pix code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/9f8f61e5a375c2e01c5187d093ce9c2409f409b0
base_folder = "/content/OCT2Hist-UseModel/pytorch-CycleGAN-and-pix2pix"

# Private module to wrapp around using cmd
def _run_subprocess(cmd):
    result = subprocess.run([cmd], capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"{cmd}")
        print(f"Command failed with exit code {result.returncode}.")
        print("Error message:")
        print(result.stderr)
        raise RuntimeError("See error")

# Run this function to set up the Neural Network with Pre-trained generator network
# path_to_generaor_network - path to generator network (*.pth file), default value is the OCT2Hist network
# Set up environment load the network parameters, run this code once
def setup_network(
    path_to_generaor_network = "/content/drive/Shareddrives/Yolab - Current Projects/_Datasets/2020-11-10 10x Model (Paper V2)/latest_net_G.pth"
    ):
    
    # Install dependencies
    _run_subprocess(f'pip install -r {base_folder}/requirements.txt')
    
    # Mount google drive (if not already mounted)
    drive.mount('/content/drive/')

    # Copy model parameters to the correct location
    _run_subprocess(f'mkdir {base_folder}/checkpoints')
    _run_subprocess(f'mkdir {base_folder}/checkpoints/pix2pix/')
    _run_subprocess(f'cp "{path_to_generaor_network}" {base_folder}/checkpoints/pix2pix/latest_net_G.pth')

# This function evaluates the neural network on input image
# Inputs:
#   im - input image (input domain, e.g. OCT) in cv format (256x256x3). Input image should be masked and cropped.
# Outputs:
#   output image (in target domain, e.g. virtual histology) in cv format
def run_network (im):
    
    # Input check
    if im.shape[:2] != (256, 256):
        raise ValueError("Image size must be 256x256 pixels to run model on.")

    # Pix2Pix implementation expects 256x512 image. Pad with zeros
    padded = np.zeros([256,512,3], np.uint8)
    padded[:,:256,:] = im[:,:,:]
    
    # Write input image in the right folder structure
    images_dir = f"{base_folder}/dataset/test"
    os.makedirs(images_dir)
    im_input_path = os.path.join(images_dir,"im1.jpg")
    cv2.imwrite(im_input_path, padded)
    
    # Run pix2pix
    _run_subprocess(f'python {base_folder}/test.py --netG resnet_9blocks --dataroot "{base_folder}/dataset/"  --model pix2pix --name pix2pix --checkpoints_dir "{base_folder}/checkpoints" --results_dir "{base_folder}/results"')

    # Load output image
    im_output_path = f"{base_folder}/results/pix2pix/test_latest/images/im1_fake_B.png"
    im_output = cv2.imread(im_output_path)
    
    return(im_output)
    