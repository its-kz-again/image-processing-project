import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from scipy.signal import correlate2d


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


# for rgb image
def correlate3d(im, kernel):
    """
    Correlate im with kernel, over all three colour channels
    """
    ims = []
    for d in range(3):
        im_corr_d = correlate2d(im[:, :, d], kernel, mode="same", boundary="symm")
        ims.append(im_corr_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


def create_mean_filter(size):
    kernel = np.ones((size, size))
    kernel /= np.sum(kernel)

    return kernel


# input image
image_name = input("please input image name(example : 1.JPG):")
size = int(input("size of kernel:"))
image = plt.imread(image_name)

kernel = create_mean_filter(size)

convolve_image = convolve3d(image, kernel)
correlate_image = correlate3d(image, kernel)

# plot image
plt.figure(figsize=(19, 16))
plt.subplot(131), plt.imshow(image), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(convolve_image), plt.title('convolve image with {0} * {1} mean filter'.format(size, size))
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(correlate_image), plt.title('correlate image with {0} * {1} mean filter'.format(size, size))
plt.xticks([]), plt.yticks([])
plt.show()