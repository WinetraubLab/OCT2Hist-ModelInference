import numpy as np;
import cv2;

from utils.show_images import readImgByPath
from copy import deepcopy

#high level masking description:
# (2) "Blackout" everything way above and way below the tissue-gel interface
# The idea is that anything that is much higher than the tissue-gel interface is not relevant because it is deep in the gel, and anything way below the interface is also not relevant because it is deep in the tissue and we don't get OCT signal that deep.

# The sub steps of this are:
# (a) We manually find the average pixel depth of the tissue-gel interface. This is not an accurate step, we basically use a mouse to click on the image where we think the average is. We call this depth Z
# (b) We black out anything above Z-100 pixels (in practice this is Z-100 microns)
# (c) We black out anything below Z+500 pixels. (in practice this is Z+500 microns)

# Main function, input is an OCT image, output is filtered image.
# Inputs:
#   img - input image, this code treats (0,0,0) as NaN.
#   min_signal_threshold (optional) - manualy set the threshold underwhich we mask image. If set to nan will apply an algorithm to compute threshold.
# Outputs:
#   img - filtered input image.
#   boolean_mask - set to true for all pixels kept, false for all pixels removed (set to 0).
#                  to apply boolean mask use (boolean_mask * img) where img is n by m by 3 matrix.
#   filter_top_bottom, min_signal_threshold are keped for debug purposes.
def mask_image(img, min_signal_threshold=np.nan):

  # Input checks and input image conversion
  assert(img.dtype == np.uint8)
  float_img = img.astype(np.float64)/255.0
  float_img[img==0] = np.nan # Treat 0 as NaN

  # We smooth input image and compute the filter on the smooth version to prevent sharp edges
  filt_img = smooth(float_img)

  # Areas with low OCT signal usually don't have any useful information, we find a threshold
  # and filter out the image below it (usually at the bottom of the image)
  if np.isnan(min_signal_threshold):
    min_signal_threshold = find_min_signal(filt_img)
  filt_img[filt_img < min_signal_threshold] = 0

  # Filtering out the gel is usful since we don't care about the gel area for histology
  filt_img, filter_top_bottom = blackout_out_of_tissue_gel(filt_img, float_img)

  # Extract the bollean mask
  boolean_mask = ~((filt_img == 0.0) | np.isnan(filt_img))

  # Apply filter on original image and convert to output format
  float_img = float_img * boolean_mask
  img = (float_img*255).astype(np.uint8)
  return img, boolean_mask, filter_top_bottom, min_signal_threshold


def get_first_zero_and_next_non_zero_idx(arr):
  """
  For an input array <arr>, returns the first zero index i_0, and the next non-zero index i_1 > i_0.
  """
  first_zero = (arr==0).argmax()
  tmp = np.copy(arr);
  tmp[:first_zero] = 0
  next_non_zero = (tmp>0).argmax(axis=0)
  return first_zero, next_non_zero


def find_the_longest_non_zero_row(m_mean_arr):
  current_count = 0
  longest_count = 0
  longest_count_start = -1
  longest_count_end = -1
  current_start = 0
  current_end = 0

  for i in range(len(m_mean_arr)):
    if m_mean_arr[i]>0:
      if current_count == 0:
        current_start = i
      current_count += 1

    if m_mean_arr[i]==0 and current_count > 0:
      current_end = i
      if current_count > longest_count:
        longest_count_start = current_start
        longest_count_end = current_end
        longest_count = current_count
        current_count = 0

  return longest_count_start, longest_count_end



def blackout_out_of_tissue_gel(filt_img, img, top_bottom_10percent_assumption = False):
  if top_bottom_10percent_assumption:
    blackout_10percent(filt_img)
  # get mean over x axis (rows) to get one value for each depth, for new thresholded image.
  m_mean = np.nanmean(filt_img, axis=1)
  m_mean_arr = np.copy(m_mean[:, 0])
  begin,end = find_the_longest_non_zero_row(m_mean_arr)
  mid = int((begin+end)/2)
  # filt_img[:start] = 0
  filter_copy = deepcopy(filt_img)
  #prepare filter, for the lower half (row > mid), and below signal (filt_img == 0).
  filt_img[:mid,:,:] = 1
  return filt_img, filter_copy


def blackout_10percent(filt_img):
  # Assuming the top and bottom 10% of the mask should be black:
  height = filt_img.shape[0]
  p = 0.1
  filt_img[0:int(p * height), :] = 0
  filt_img[int((1 - p) * height):, :] = 0


def find_min_signal(filt_img):
  m_mean_max, m_mean_min = get_rows_min_max(filt_img)
  # Finally we define a threshold for OCT intensity, anything below that will be blacked out
  minSignal = 0.28 * (m_mean_max - m_mean_min) + m_mean_min
  return minSignal


def get_rows_min_max(filt_img):
  # Average over x axis (rows) to get one value for each depth
  m_mean = np.nanmean(filt_img[:,:,0], axis=1)
  # Then we figure out what is the "brightest" row by taking percentile:
  m_mean_max = np.percentile(m_mean, 99, axis=0)
  # Then we figure out what is the noise floor of the device, by examining the bottom 50 rows of OCT image
  m_mean_min = np.mean(m_mean[-50:])
  return m_mean_max, m_mean_min


def smooth(img):
  # Apply a gaussian filter to smooth everything, it will help make the thresholding smoother
  sigma = 20
  # the default filter size in Matlab
  filter_size = int(2 * np.ceil(2 * sigma) + 1)
  filt_img = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
  return filt_img


def oct_get_image_and_preprocess(oct_input_image_path):
  # Path to an OCT image to convert
  oct_input_image_path = "/content/drive/Shareddrives/Yolab - Current Projects/_Datasets/2020-11-10 10x Raw Data Used In Paper (Paper V2)/LG-19 - Slide04_Section02 (Fig 3.c)/OCTAligned.tiff"

  # how many microns per pixel for each axis
  microns_per_pixel_z = 1
  microns_per_pixel_x = 1

  # Path to a folder in drive to output the converted H&E images, leave blank if
  # you don't want to save H&E image to drive.
  histology_output_image_folder = ""




if __name__ == '__main__':
  o2h_input = oct_get_image_and_preprocess("/Users/dannybarash/Code/oct/OCT2Hist-UseModel/baseline_input.tiff")

  oct_input_image_path = "/Users/dannybarash/Code/oct/OCT2Hist-UseModel/baseline_input.tiff"
  oct_image_orig = cv2.imread(oct_input_image_path)
  oct_image_orig = cv2.cvtColor(oct_image_orig, cv2.COLOR_BGR2RGB)

  masked_image, mask = mask_image(oct_image_orig)
  #make it boolean
  mask[(mask > 0)] = 1
  #reverse colors
  mask = 1- mask
  mask_path = "/Users/dannybarash/Code/oct/OCT2Hist-UseModel/baseline_mask.tiff"
  gt_mask = cv2.imread(mask_path)
  gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
