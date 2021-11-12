import numpy as np
import matplotlib.pylab as plt
import cv2


# input image
image_name = input("please input image name(example : 1.JPG):")
size = int(input("size of kernel:"))
image = plt.imread(image_name)


# -----------------------------------------------------------------
# method1
# use open-cv library and GaussianBlur function
res = cv2.GaussianBlur(image, (size, size), 0)
# -----------------------------------------------------------------


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(image), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res), plt.title('{0} * {1} gaussian filtering'.format(size, size))
plt.xticks([]), plt.yticks([])
plt.show()