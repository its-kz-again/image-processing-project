import numpy as np
import matplotlib.pylab as plt
from PIL import ImageFilter, Image
from scipy import stats


# mode filter method1 - for rgb image - slow method
def mode_filter_method1(img, size_r, size_c):
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
                l = stats.mode(k, axis=None)
                im[i, j] = l[0][0]

        ims.append(im)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv


image_name = input("please input image name(example : 1.JPG):")
size = int(input("size of kernel:"))

# -----------------------------------------------------------------
# method1
# If you want to use the method1, uncomment on the following line
# for industrial images and high quality images don't use this method
# because this method is very slow
# image = plt.imread(image_name)
# res = mode_filter_method1(image, size, size)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# method2
# faster than method1
# use PIL library - for rgb image
image = Image.open(image_name)
res = image.filter(ImageFilter.ModeFilter(size=size))
# -----------------------------------------------------------------


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(image), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res), plt.title('{0} * {1} mode filtering'.format(size, size))
plt.xticks([]), plt.yticks([])
plt.show()