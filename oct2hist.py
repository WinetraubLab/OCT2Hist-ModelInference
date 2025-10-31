from utils.show_images import *
from utils.masking import mask_image
from utils.gray_level_rescale import gray_level_rescale
import utils.pix2pix as pix2pix
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Run this function to set up the Neural Network with Pre-trained oct2hist generator network
def setup_network():
  pix2pix.setup_network("/content/drive/Shareddrives/Yolab - Current Projects/_Datasets/2020-11-10 10x OCT2Hist Model (Paper V2)/latest_net_G.pth","oct2hist")

# This function evaluates the neural network on input image
# Inputs:
#   oct_image - input oct image in gray scale (256x256x3). Input image should be scanned with 10x lens and z-stacked
#   microns_per_pixel_x - how many microns is each pixel on x direction (lateral direction). This is determined by B-Scan parameters, not the lens.
#   microns_per_pixel_z - how many microns is each pixel on z direction (axial direction). This is determined by spectrumeter width not light source FWHM.
# Preprocessing configuration. Set this parameters to false if you would like to skip them
#   apply_masking - should we perform the mask step?
#   min_signal_threshold - By default this is NaN, set to numeric value if you would like to apply a min threshold for masking rather than use algorithm. Good value to use is 0.1
#   apply_gray_level_scaling - should we rescale gray level to take full advantage of dynamic range?
#   appy_resolution_matching - should we match resolution to the trained images?
# Outputs:
#   output image (in target domain, e.g. virtual histology) in RGB format
#   masked_image - if apply_masking=true, otherwise it will be identical to im 
#   network_input_image - the image that is loaded to the network
def run_network (oct_image, 
                microns_per_pixel_x=1,
                microns_per_pixel_z=1,
                apply_masking=True,
                min_signal_threshold=np.nan,
                apply_gray_level_scaling=True,
                appy_resolution_matching=True,
                ):
  # Mask
  if apply_masking:
    masked_image, *_ = mask_image(oct_image, min_signal_threshold=min_signal_threshold)    
  else:
    masked_image = oct_image

  if apply_gray_level_scaling:
    rescaled_image = gray_level_rescale(masked_image)
  else:
    rescaled_image = masked_image

  # Apply resolution matching
  original_height, original_width = rescaled_image.shape[:2]
  if appy_resolution_matching:
    target_width = int(original_width / 2)
    target_height = int(original_height / 2)
    # Apply the resolution change
    o2h_input = cv2.resize(rescaled_image, [target_width,target_height] , interpolation=cv2.INTER_AREA)
  else:
    o2h_input = rescaled_image

  # Run the neural net
  virtual_histology_image = pix2pix.run_network(o2h_input,"oct2hist")

  # Post process, return image to original size
  virtual_histology_image_resized = cv2.resize(virtual_histology_image, [original_width,original_height] , interpolation=cv2.INTER_AREA)

  return virtual_histology_image_resized, masked_image, o2h_input
    
