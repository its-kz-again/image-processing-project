import numpy as np
import matplotlib.pylab as plt
import cv2


# contraharmonic mean filter
def contraharmonic_mean(im, size, Q):
    num = np.power(im, Q + 1)
    denom = np.power(im, Q)
    kernel = np.full(size, 1.0)
    result = cv2.filter2D(num, -1, kernel) / (cv2.filter2D(denom, -1, kernel) + 1) # plus one for avoid zero dividing
    return result


# create salt_pepper noise
def salt_pepper(im, prob):
    # Extract image dimensions
    row, col = im.shape

    # Declare salt & pepper noise ratio
    s_vs_p = 0.5
    output = np.copy(im)

    # Apply salt noise on each pixel individually
    num_salt = np.ceil(prob * im.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in im.shape]
    output[tuple(coords)] = 1

    # Apply pepper noise on each pixel individually
    num_pepper = np.ceil(prob * im.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in im.shape]
    output[tuple(coords)] = 0

    return output


# input gray scale image
image_name = input("please input image name(example : 1.JPG):")
image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

# Call salt & pepper function with probability = 0.5
sp_05 = salt_pepper(image, 0.5)

contraharmonic_mean_image1 = contraharmonic_mean(sp_05, (3, 3), 0.5)
contraharmonic_mean_image2 = contraharmonic_mean(sp_05, (3, 3), 1.5)


# plot image
plt.figure(figsize=(19, 16))
plt.subplot(141), plt.imshow(image, cmap='gray'), plt.title('original gray scale image')
plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(sp_05, cmap='gray'), plt.title('noisy image')
plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(contraharmonic_mean_image1, cmap='gray'), plt.title('3 * 3 contraharmonic mean filtering - Q = 0.5')
plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(contraharmonic_mean_image2, cmap='gray'), plt.title('3 * 3 contraharmonic mean filtering - Q = 1.5')
plt.xticks([]), plt.yticks([])
plt.show()
