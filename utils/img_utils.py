import cv2
import matplotlib.pyplot as plt


def showImgByPath(path):
  """Show the image for filepath <path>"""
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.figure()
  plt.imshow(image)

def readImgByPath(path):
  """Return the image for filepath <path> """
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def showImg(image):
  """Show the image, which content is in image."""
  plt.figure()
  plt.imshow(image)

def showTwoImgs(img1, img2):
  """Show both images, side by side."""
  plt.subplot(1,2,1);
  plt.imshow(img1);
  plt.subplot(1,2,2);
  plt.imshow(img2);

