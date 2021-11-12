import numpy as np
import matplotlib.pylab as plt
import cv2
from scipy.signal import convolve2d
from skimage import filters


# sobel filter method1 for rgb image
def sobel_filter_method1(im, sx, sy):
    ims = []
    for d in range(3):
        gx = convolve2d(im[:, :, d], sx, mode="same", boundary="symm")
        gy = convolve2d(im[:, :, d], sy, mode="same", boundary="symm")
        ims.append(np.sqrt(gx * gx + gy * gy))

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


# sobel filter method2 for rgb image
def sobel_filter_method2(im):
    ims = []
    for d in range(3):
        edge_h = filters.sobel_h(im[:, :, d])
        edge_v = filters.sobel_v(im[:, :, d])
        edge = np.sqrt(edge_h ** 2 + edge_v ** 2)
        edge = edge / np.max(edge) * 255
        ims.append(edge)

    im_sobel = np.stack(ims, axis=2).astype("uint8")

    return im_sobel


# sobel filter method3
def sobel_filter_method3(im):
    sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)  # x-order = 1 , y-order = 0 =>> sobel_x
    sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)  # x-order = 1 , y-order = 0 =>> sobel_x
    sobel = (sobelx ** 2 + sobely ** 2) ** 0.5

    return sobel


# 15 * 15 gaussian filter + sobel filter method3
def sobel_filter_method3_with_gaussian(im):
    gaussian = cv2.GaussianBlur(im, (15, 15), 0)
    sobelx = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)  # x-order = 1 , y-order = 0 =>> sobel_x
    sobely = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=3)  # x-order = 1 , y-order = 0 =>> sobel_x
    sobel = (sobelx ** 2 + sobely ** 2) ** 0.5

    return sobel

# input image
image_name = input("please input image name(example : 1.JPG):")

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # vertical Mask of Sobel Operator
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # horizontal Mask of Sobel Operator

# -----------------------------------------------------------------
# method1
# If you want to use the method1, uncomment on the following line
# rgb_image = plt.imread(image_name)
# sobel_image = sobel_filter_method1(rgb_image, sobel_x, sobel_y)
# plt.figure(figsize=(19, 16))
# plt.subplot(121), plt.imshow(rgb_image), plt.title('original image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(sobel_image), plt.title('sobel filter for rgb image')
# plt.xticks([]), plt.yticks([])
# plt.show()
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method2
# If you want to use the method2, uncomment on the following line
# rgb_image = plt.imread(image_name)
# sobel_image2 = sobel_filter_method2(rgb_image)
# plt.figure(figsize=(19, 16))
# plt.subplot(121), plt.imshow(rgb_image), plt.title('original image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(sobel_image2), plt.title('sobel filter for rgb image')
# plt.xticks([]), plt.yticks([])
# plt.show()
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method3 for gray scale image
# use open-cv library and Sobel function
gray_image = cv2.imread(image_name, 0)
res = sobel_filter_method3(gray_image)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 15 * 15 gaussian filter + method3 for gray scale image
res1 = sobel_filter_method3_with_gaussian(gray_image)
# -----------------------------------------------------------------

# plot image
plt.figure(figsize=(19, 16))
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res, cmap='gray'), plt.title('sobel filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res1, cmap='gray'), plt.title('15 * 15 gaussian + sobel filter')
plt.xticks([]), plt.yticks([])
plt.show()
