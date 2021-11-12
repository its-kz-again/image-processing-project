import numpy as np
import matplotlib.pylab as plt
import cv2

# input image
image_name = input("please input image name(example : 1.JPG):")
gray_image = cv2.imread(image_name, 0)

blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

wide = cv2.Canny(blurred, 10, 180)  # low = 10, high = 180
mid = cv2.Canny(blurred, 30, 100)   # low = 30, high = 100
tight = cv2.Canny(blurred, 180, 200)  # low = 180, high = 200


# plot
plt.figure(figsize=(19, 16))
plt.imshow(gray_image, cmap='gray')
plt.title('original gray scale image')
plt.axis('off')
plt.show()
# canny filters image
plt.figure(figsize=(20, 16))
plt.subplot(131), plt.imshow(wide, cmap='gray'), plt.title('wide canny filter')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(mid, cmap='gray'), plt.title('mid canny filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(tight, cmap='gray'), plt.title('tight canny filter')
plt.xticks([]), plt.yticks([])
plt.show()