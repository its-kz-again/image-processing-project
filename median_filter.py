import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import median_filter
import cv2


# median filter method1 - for rgb image - slow method
def median_filter_method1(img, size_r, size_c):
    r = img.shape[0] + size_r - 1
    c = img.shape[1] + size_c - 1
    z = np.zeros((r, c))
    ims = []

    # 3 channel of rgb image
    for d in range(3):
        im = img[:, :, d].copy()
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                z[i + np.int((size_r - 1) / 2), j + np.int((size_c - 1) / 2)] = im[i, j]

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                k = z[i:i + size_r, j:j + size_c]
                l = np.median(k)
                im[i, j] = l

        ims.append(im)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


# median filter method2 with scipy library - for rgb image
def median_filter_method2(im, size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im[:, :, d], size=(size, size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


# input image
image_name = input("please input image name(example : 1.JPG):")
size = int(input("size of kernel:"))
image = plt.imread(image_name)

# -----------------------------------------------------------------
# method1
# If you want to use the method1, uncomment on the following line
# for industrial images and high quality images don't use this method
# because this method is very slow
# res = median_filter_method1(image, size, size)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# method2
# faster than method1
# If you want to use the method1, uncomment on the following line
# res = median_filter_method2(image, size)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# method3
# use open-cv library and medianBlur function - faster than method2
res = cv2.medianBlur(image, size)
# -----------------------------------------------------------------


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(image), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res), plt.title('{0} * {1} median filtering'.format(size, size))
plt.xticks([]), plt.yticks([])
plt.show()