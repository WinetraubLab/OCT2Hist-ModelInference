import numpy as np;
import cv2;

from utils.img_utils import readImgByPath


def get_first_zero_and_next_non_zero_idx(arr):
  """
  For an input array <arr>, returns the first zero index i_0, and the next non-zero index i_1 > i_0.
  """
  first_zero = (arr==0).argmax()
  tmp = np.copy(arr);
  tmp[:first_zero] = 0
  next_non_zero = (tmp>0).argmax(axis=0)
  return first_zero, next_non_zero

def mask_image(oct_input_image_path):
  # a new image from
  path_new_image = oct_input_image_path
  img = readImgByPath(path_new_image)

  filt_img = smooth(img)

  filt_below_min_signal(filt_img)

  # height, mid = blackout_out_of_tissue_gel(filt_img, img)
  img = blackout_out_of_tissue_gel(filt_img, img)
  #
  # top_row = max(mid - 128, 0)
  # bottom_row = min(mid + 128, height)
  # cropped_img = img[top_row:bottom_row, :, :]


  return img, mid


def blackout_out_of_tissue_gel(filt_img, img):
  # Assuming the top and bottom 10% of the mask should be black:
  height = filt_img.shape[0]
  p = 0.1
  filt_img[0:int(p * height), :] = 0
  filt_img[int((1 - p) * height):, :] = 0
  # get mean over x axis (rows) to get one value for each depth, for new thresholded image.
  m_mean = np.nanmean(filt_img, axis=1);
  # assume there's a non-black area around the middle of the image, surrounded by black area.
  # Find the first black line, and the next non black line.
  m_mean_arr = np.copy(m_mean[:, 0])
  first_zero, next_non_zero = get_first_zero_and_next_non_zero_idx(m_mean_arr);
  # Do the same to the vertically flipped (mirrored around the x axis) image.
  flipped = np.copy(np.flip(m_mean_arr))
  first_zero_from_the_end, next_non_zero_from_the_end = get_first_zero_and_next_non_zero_idx(flipped);
  h = len(flipped)
  first_zero_from_the_end = h - first_zero_from_the_end
  next_non_zero_from_the_end = h - next_non_zero_from_the_end
  # "Blackout" everything above and below the tissue-gel interface
  margin = 10;  # margin around the high-enough-snr area.
  top = next_non_zero - margin;
  bottom = next_non_zero_from_the_end + margin;
  img[:top, :] = 0
  img[bottom:, :] = 0
  mid = int((bottom + top) / 2.0);
  filt_img[:mid, :] = 1;
  img[filt_img == 0] = 0;
  return img


def filt_below_min_signal(filt_img):
  m_mean_max, m_mean_min = get_rows_min_max(filt_img)
  # Finally we define a threshold for OCT intensity, anything below that will be blacked out
  minSignal = 0.28 * (m_mean_max - m_mean_min) + m_mean_min;
  filt_img[filt_img < minSignal] = 0;


def get_rows_min_max(filt_img):
  # Average over x axis (rows) to get one value for each depth
  m_mean = np.nanmean(filt_img, axis=1)
  # Then we figure out what is the "brightest" row by taking percentile:
  m_mean_max = np.percentile(m_mean, 99, axis=0)
  # Then we figure out what is the noise floor of the device, by examining the bottom 50 rows of OCT image
  m_mean_min = np.mean(m_mean[-50:]);
  return m_mean_max, m_mean_min


def smooth(img):
  # Apply a gaussian filter to smooth everything, it will help make the thresholding smoother
  sigma = 20
  # the default filter size in Matlab
  filter_size = int(2 * np.ceil(2 * sigma) + 1)
  filt_img = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
  return filt_img
