import numpy as np
import matplotlib.pylab as plt
import cv2


# with numpy
def fourier_transform_method1(im):
    f = np.fft.fft2(im)
    fshift = np.fft.fftshift(f)
    # Magnitude of the function is 20.log(abs(f))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum


# with open-cv
def fourier_transform_method2(im):
    dft = cv2.dft(np.float32(im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


# input image
image_name = input("please input image name(example : 1.JPG):")
gray_image = cv2.imread(image_name, 0)

# -----------------------------------------------------------------
# method1
# fourier transform with numpy
# If you want to use this method, uncomment on the following line
# magnitude_spectrum = fourier_transform_method1(gray_image)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# method2
# fourier transform with open-cv
magnitude_spectrum = fourier_transform_method2(gray_image)
# -----------------------------------------------------------------


# plot
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(gray_image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
