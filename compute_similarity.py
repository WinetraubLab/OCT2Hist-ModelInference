# This module computes similarity between two images for a given resolution

import cv2
import numpy as np
from skimage import color
from skimage.metrics import structural_similarity as ssim

# Compute similarity between im1 and im2 (load using openCV).
# We can add blur radius for gaussian bluring (pixels)
# Returns ssim
def compute_similarity (im1, im2, blur_radius=0):
  # Function to calculate SSIM between two images
  def calculate_ssim(image1, image2):
      # Convert images to grayscale
      gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  
      # Compute SSIM
      score, _ = ssim(gray1, gray2, full=True)
      return score
    
  # Blur the images
  blurred_im1 = blur_image(im1, blur_radius)
  blurred_im2 = blur_image(im2, blur_radius)

  # Compute simularity
  return calculate_ssim(blurred_im1,blurred_im2)

# Blur image using gauissian filter
def blur_image(image, blur_radius):
  if blur_radius>0:
      sigma = blur_radius
      filter_size = int(2 * np.ceil(2 * sigma) + 1) # the default filter size in Matlab
      filt_img = cv2.GaussianBlur(image, (filter_size, filter_size), sigma)
      return filt_img
  else:
    return image
