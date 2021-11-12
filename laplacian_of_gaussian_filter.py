import numpy as np
import matplotlib.pylab as plt
import cv2

# input image
image_name = input("please input image name(example : 1.JPG):")
gray_image = cv2.imread(image_name, 0)

# reduce noise
blur = cv2.GaussianBlur(gray_image, (15, 15), 0)

filtered_image = cv2.Laplacian(blur, ksize=5, ddepth=cv2.CV_64F)
filtered_image = filtered_image / filtered_image.max() * 255

filtered_image1 = np.copy(filtered_image)
filtered_image2 = np.copy(filtered_image)
# method1
filtered_image1[filtered_image1 < 0] = 0
# method2
filtered_image2 = cv2.convertScaleAbs(filtered_image2)

# plot
plt.figure(figsize=(19, 16))
plt.imshow(gray_image, cmap='gray')
plt.title('original gray scale image')
plt.axis('off')
plt.show()
# plot 2 methods of laplacian of gaussian
for i in ([[filtered_image1, 'laplacian of gaussian method1'], [filtered_image2, 'laplacian of gaussian method2']]):
    plt.imshow(i[0], cmap='gray')
    plt.title(i[1])
    plt.axis('off')
    plt.show()

