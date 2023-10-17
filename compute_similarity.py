# This module computes similarity between two images for a given resolution

import cv2
from skimage import color
from skimage.metrics import structural_similarity as ssim

# Compute similarity between im1 and im2 (load using openCV).
# We can add blur radius for gaussian bluring (pixels)
# Returns ssim
def compute_similarity (im1, im2, blur_radius=0):
  
  def blur_image(image, blur_radius):
    return cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

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
