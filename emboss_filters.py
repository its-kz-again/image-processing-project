import numpy as np
import matplotlib.pylab as plt
import cv2


# input image
image_name = input("please input image name(example : 1.JPG):")
image = plt.imread(image_name)

# for rgb image
kernel = np.array([[-2, -1, 0], [-1,  1, 1], [0,  1, 2]])
emboss_image = cv2.filter2D(image, -1, kernel)
# plot image
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(image), plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(emboss_image), plt.title('emboss image')
plt.xticks([]), plt.yticks([])
plt.show()

# for edge detection and emboss edges
gray_image = cv2.imread(image_name, 0)
kernel2 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
kernel3 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
emboss_image1 = cv2.filter2D(gray_image, -1, kernel2)
emboss_image2 = cv2.filter2D(gray_image, -1, kernel3)
emboss_image3 = emboss_image1 + emboss_image2

# plot image
plt.figure(figsize=(19, 16))
plt.subplot(141), plt.imshow(gray_image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(emboss_image1, cmap='gray'), plt.title('emboss image with M1 kernel')
plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(emboss_image2, cmap='gray'), plt.title('emboss image with M2 kernel')
plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(emboss_image3, cmap='gray'), plt.title('emboss image1 + emboss image2')
plt.xticks([]), plt.yticks([])
plt.show()