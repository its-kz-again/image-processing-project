import numpy as np
import matplotlib.pylab as plt
import cv2
from skimage import io, filters
from scipy import ndimage


# robert filter method1 for gray scale image
def robert_filter_method1(im, RCV, RCH):
    v = ndimage.convolve(im, RCV)
    h = ndimage.convolve(im, RCH)
    edge = np.sqrt(v ** 2 + h ** 2)

    return edge


# 15 * 15 gaussian filter + robert filter method1
def robert_filter_method1_with_gaussian(im, RCV, RCH):
    gaussian = cv2.GaussianBlur(im, (15, 15), 0)
    v = ndimage.convolve(gaussian, RCV)
    h = ndimage.convolve(gaussian, RCH)
    edge = np.sqrt(v ** 2 + h ** 2)

    return edge


# input image
image_name = input("please input image name(example : 1.JPG):")
# read gray scale image with skiamge library
image = io.imread(image_name)
image = image.astype('float64')
gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
gray_image /= 255


RCV = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])  # robert cross vertical
RCH = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])  # robert cross horizontal

# -----------------------------------------------------------------
# method1 for gray scale image
res = robert_filter_method1(gray_image, RCV, RCH)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 15 * 15 gaussian filter + method1 for gray scale image
res1 = robert_filter_method1_with_gaussian(gray_image, RCV, RCH)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# robert filter method2 use skimage.filters
# If you want to use the method1, uncomment on the following line
# res2 = filters.roberts(gray_image)
# plt.imshow(res2, cmap=plt.get_cmap('gray'))
# plt.title('robert filter with skiamge.filters')
# plt.axis('off')
# plt.show()
# -----------------------------------------------------------------


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res, cmap='gray'), plt.title('robert filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res1, cmap='gray'), plt.title('15 * 15 gaussian + robert filter')
plt.xticks([]), plt.yticks([])
plt.show()
