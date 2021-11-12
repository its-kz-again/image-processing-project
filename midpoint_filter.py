import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import maximum_filter, minimum_filter
import cv2


# 3 * 3 midpoint filter
def midpoint_filter(im):
    maxf = maximum_filter(im, (3, 3))
    minf = minimum_filter(im, (3, 3))
    midpoint = (maxf + minf) / 2

    return midpoint


# create salt_pepper noise
def salt_pepper(im, prob):
    # Extract image dimensions
    row, col = im.shape

    # Declare salt & pepper noise ratio
    s_vs_p = 0.5
    output = np.copy(im)

    # Apply salt noise on each pixel individually
    num_salt = np.ceil(prob * im.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in im.shape]
    output[tuple(coords)] = 1

    # Apply pepper noise on each pixel individually
    num_pepper = np.ceil(prob * im.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in im.shape]
    output[tuple(coords)] = 0

    return output


# input gray scale image
image_name = input("please input image name(example : 1.JPG):")
image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

# Call salt & pepper function with probability = 0.5
sp_05 = salt_pepper(image, 0.5)

midpoint_image = midpoint_filter(sp_05)


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(sp_05, cmap='gray'), plt.title('noisy image')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(midpoint_image, cmap='gray'), plt.title('3 * 3 midpoint filtering')
plt.xticks([]), plt.yticks([])
plt.show()
