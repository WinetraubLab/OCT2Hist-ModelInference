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

def showThreeImgs(image1, image2, image3):
  # Create a figure with three subplots
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  # Display the first image in the first subplot
  axes[0].imshow(image1)  # Change 'gray' to the appropriate colormap
  axes[0].set_title('Image 1')
  
  # Display the second image in the second subplot
  axes[1].imshow(image2)  # Change 'gray' to the appropriate colormap
  axes[1].set_title('Image 2')
  
  # Display the third image in the third subplot
  axes[2].imshow(image3)  # Change 'gray' to the appropriate colormap
  axes[2].set_title('Image 3')
  
  # Adjust spacing between subplots
  plt.tight_layout()
  
  # Show the figure
  plt.show()
