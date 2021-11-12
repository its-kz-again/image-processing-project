import numpy as np
import matplotlib.pylab as plt
import cv2


# scharr filter
def scharr_filter_method1(im):
    scharrx = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    scharr = (scharrx ** 2 + scharry ** 2) ** 0.5

    return scharr


# 15 * 15 gaussian filter + scharr filter
def scharr_filter_method1_with_gaussian(im):
    gaussian = cv2.GaussianBlur(im, (15, 15), 0)
    scharrx = cv2.Scharr(gaussian, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gaussian, cv2.CV_64F, 0, 1)
    scharr = (scharrx ** 2 + scharry ** 2) ** 0.5

    return scharr


# input image
image_name = input("please input image name(example : 1.JPG):")
gray_image = cv2.imread(image_name, 0)


# -----------------------------------------------------------------
# method1 for gray scale image
res = scharr_filter_method1(gray_image)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 15 * 15 gaussian filter + method1 for gray scale image
res1 = scharr_filter_method1_with_gaussian(gray_image)
# -----------------------------------------------------------------


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res, cmap='gray'), plt.title('scharr filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res1, cmap='gray'), plt.title('15 * 15 gaussian + scharr filter')
plt.xticks([]), plt.yticks([])
plt.show()