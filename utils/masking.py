import numpy as np;
import cv2;

from utils.show_images import readImgByPath


def get_first_zero_and_next_non_zero_idx(arr):
  """
  For an input array <arr>, returns the first zero index i_0, and the next non-zero index i_1 > i_0.
  """
  first_zero = (arr==0).argmax()
  tmp = np.copy(arr);
  tmp[:first_zero] = 0
  next_non_zero = (tmp>0).argmax(axis=0)
  return first_zero, next_non_zero

def mask_image(img):
  assert(img.dtype == np.uint8)
  float_img = img.astype(np.float64)/255.0
  float_img[img==0] = np.nan
  filt_img = smooth(float_img)
  min_signal = find_min_signal(filt_img)
  filt_img[filt_img < min_signal] = 0
  filt_img = blackout_out_of_tissue_gel(filt_img, float_img, min_signal)
  float_img[(filt_img == 0.0) | np.isnan(filt_img)] = 0
  img = (float_img*255).astype(np.uint8)
  return img, filt_img


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



def blackout_out_of_tissue_gel(filt_img, img, min_signal, top_bottom_10percent_assumption = False):
  if top_bottom_10percent_assumption:
    blackout_10percent(filt_img)
  # get mean over x axis (rows) to get one value for each depth, for new thresholded image.
  m_mean = np.nanmean(filt_img, axis=1)
  m_mean_arr = np.copy(m_mean[:, 0])
  begin,end = find_the_longest_non_zero_row(m_mean_arr)
  mid = int((begin+end)/2)
  #remove any line which is not part of the longest non zero sequence
  # filt_img[:start] = 0
  #apply filter on image, for the lower half (row > end), and below signal (filt_img == 0).
  filt_img[:mid,:,:] = 1
  return filt_img


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
