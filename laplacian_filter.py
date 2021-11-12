import numpy as np
import matplotlib.pylab as plt
import cv2


def laplacian_filter_method1(im, L1, L2, L3, L4, L5):
    res1 = cv2.filter2D(im, -1, L1)  # with L1 kernel
    res2 = cv2.filter2D(im, -1, L2)  # with L2 kernel
    res3 = cv2.filter2D(im, -1, L3)  # with L3 kernel
    res4 = cv2.filter2D(im, -1, L4)  # with L4 kernel
    res5 = cv2.filter2D(im, -1, L5)  # with L5 kernel

    return res1, res2, res3, res4, res5


def plot_image(im, im1, im2, im3, im4, im5):
    for i in ([im, 'Original Image'], [im1, 'laplacian1'], [im2, 'laplacian2'], [im3, 'laplacian3'], [im4, 'laplacian4'], [im5, 'excessive sharpening']):
        plt.figure(figsize=(30, 20))
        plt.imshow(i[0], cmap='gray')  # if you plot rgb image - should rewrite this code to --> plt.imshow(i[0])
        plt.title(i[1])
        plt.axis('off')
        plt.show()



# input image
image_name = input("please input image name(example : 1.JPG):")
image = plt.imread(image_name)
gray_image = cv2.imread(image_name, 0)

L1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # for edge detection
L2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # for edge detection
L3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # for sharpening
L4 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # for sharpening
L5 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])  # for more sharpening

# -------------------------------------------------------
# method without denoising
# first argument : image or gray_image - you can change the first argument
# If you want to use this method, uncomment on the following lines
# r1, r2, r3, r4, r5 = laplacian_filter_method1(gray_image, L1, L2, L3, L4, L5)
# plot_image(gray_image, r1, r2, r3, r4, r5)
# ---------------------------------------------------------

# -------------------------------------------------------
# method with denoising
# first argument : image or gray_image - you can change the first argument
denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 4, 7, 21)
r6, r7, r8, r9, r10 = laplacian_filter_method1(denoised_image, L1, L2, L3, L4, L5)
plot_image(gray_image, r6, r7, r8, r9, r10)
# ---------------------------------------------------------



