from scipy.signal import convolve2d
import numpy as np
import matplotlib.pylab as plt
import cv2

# for rgb image
def convolve3d(im, kernel):
    """
    Convolves im with kernel, over all three colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = convolve2d(im[:, :, d], kernel, mode="same", boundary="symm")
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


# mean filter with convolve3d method - for rgb image
def mean_filter_method1(im, size):
    kernel = np.ones((size, size))
    kernel /= np.sum(kernel)  # kernel = filter
    result = convolve3d(im, kernel)

    return result


# mean filter with filter2D
def mean_filter_method2(im, size):
    kernel = np.ones((size, size))
    kernel /= np.sum(kernel)  # kernel = filter
    result = cv2.filter2D(im, -1, kernel)

    return result


# input image
image_name = input("please input image name(example : 1.JPG):")
size = int(input("size of kernel:"))
image = plt.imread(image_name)

# -----------------------------------------------------------------
# method1
# If you want to use the method1, uncomment on the following line
# res = mean_filter_method1(image, size)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method2
# faster than method1
# If you want to use the method1, uncomment on the following line
# res = mean_filter_method2(image, size)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method3
# use open-cv library and blur function
res = cv2.blur(image, (size, size))
# -----------------------------------------------------------------


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(image), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res), plt.title('{0} * {1} mean filtering'.format(size, size))
plt.xticks([]), plt.yticks([])
plt.show()
