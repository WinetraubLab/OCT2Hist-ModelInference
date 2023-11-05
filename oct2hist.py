from utils.show_images import *
from utils.masking import mask_image, mask_image_gel
from utils.gray_level_rescale import gray_level_rescale
import utils.pix2pix as pix2pix
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Run this function to set up the Neural Network with Pre-trained oct2hist generator network

def setup_network():
  pix2pix.setup_network("/content/drive/Shareddrives/Yolab - Current Projects/_Datasets/2020-11-10 10x OCT2Hist Model (Paper V2)/latest_net_G.pth","oct2hist")

def run_network (oct_image,
                microns_per_pixel_x=1,
                microns_per_pixel_z=1,
                apply_masking=True,
                min_signal_threshold=np.nan,
                apply_gray_level_scaling=True,
                appy_resolution_matching=True,
                ):

  if apply_gray_level_scaling:
    oct_image = gray_level_rescale(oct_image)
  else:
    oct_image = oct_image

  # Mask
  if apply_masking:
    masked_image,  *_ = mask_image_gel(oct_image, min_signal_threshold=min_signal_threshold)
  else:
    masked_image = oct_image



  # Apply resolution matching
  original_height, original_width = masked_image.shape[:2]
  if appy_resolution_matching:
    # Compute compression ratio
    target_width = original_width * microns_per_pixel_x // 4 # Target resolution is 4 microns per pixel on x axis. We use // to round to integer
    target_height = original_height * microns_per_pixel_z // 2 # Target resolution is 2 microns per pixel on z axis. We use // to round to integer

    if target_width!=256 or target_height!=256:
      raise ValueError(f"OCT2Hist works on images which have total size of 1024 microns by 512 microns (x,z). Input oct_image has size of {original_width*microns_per_pixel_x} by {original_height*microns_per_pixel_z} microns. Please crop or pad image")

    # Apply the resolution change
    o2h_input = cv2.resize(masked_image, [target_height,target_width] , interpolation=cv2.INTER_AREA)
  else:
    o2h_input = masked_image

  # Run the neural net
  virtual_histology_image = pix2pix.run_network(o2h_input,"oct2hist")

  # Post process, return image to original size
  virtual_histology_image_resized = cv2.resize(virtual_histology_image, [original_width,original_height] , interpolation=cv2.INTER_AREA)

  return virtual_histology_image_resized, masked_image, o2h_input

