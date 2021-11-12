import numpy as np
import matplotlib.pylab as plt
import cv2
from skimage import io, filters


# prewitt filter method1 for rgb image
def prewitt_filter_method1(im, px, py):
    prewittx = cv2.filter2D(im, -1, px)
    prewitty = cv2.filter2D(im, -1, py)
    prewitt = prewittx + prewitty

    return prewitt


# prewitt filter method2
def prewitt_filter_method2(im, px, py):
    prewittx = cv2.filter2D(im, -1, px)
    prewitty = cv2.filter2D(im, -1, py)
    prewitt = (prewittx ** 2 + prewitty ** 2) ** 0.5

    return prewitt


# 15 * 15 gaussian filter + prewitt filter method2
def prewitt_filter_method2_with_gaussian(im, px, py):
    gaussian = cv2.GaussianBlur(im, (15, 15), 0)
    prewittx = cv2.filter2D(gaussian, -1, px)
    prewitty = cv2.filter2D(gaussian, -1, py)
    prewitt = (prewittx ** 2 + prewitty ** 2) ** 0.5

    return prewitt

# input image
image_name = input("please input image name(example : 1.JPG):")

# correlation kernel
py = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
px = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])


# -----------------------------------------------------------------
# method1
# If you want to use the method1, uncomment on the following line
# rgb_image = plt.imread(image_name)
# prewitt_image = prewitt_filter_method1(rgb_image, px, py)
# plt.figure(figsize=(19, 16))
# plt.subplot(121), plt.imshow(rgb_image), plt.title('original image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(prewitt_image), plt.title('prewitt filter for rgb image')
# plt.xticks([]), plt.yticks([])
# plt.show()
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# method2 for gray scale image
# use open-cv library and filter2D function
image = io.imread(image_name)
image = image.astype('float64')
gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
gray_image /= 255
res = prewitt_filter_method2(gray_image, px, py)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 15 * 15 gaussian filter + method2 for gray scale image
res1 = prewitt_filter_method2_with_gaussian(gray_image, px, py)
# -----------------------------------------------------------------

# plot image
plt.figure(figsize=(19, 16))
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res, cmap='gray'), plt.title('prewitt filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res1, cmap='gray'), plt.title('15 * 15 gaussian + prewitt filter')
plt.xticks([]), plt.yticks([])
plt.show()


