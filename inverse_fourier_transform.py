import numpy as np
import matplotlib.pylab as plt
import cv2


# with numpy
def inverse_fourier_transform_method1(im):
    # fourier transform
    f = np.fft.fft2(im)
    fshift = np.fft.fftshift(f)
    # If you do not want to filter and only want the original image recovered
    # than comment the code for frequency domain filtering
    # ---------------------------------------------------------------
    # frequency domain filtering - ideal high pass filter
    rows, cols = im.shape
    r, c = rows // 2, cols // 2
    fshift[r - 30:r + 31, c - 30:c + 31] = 0  # it means that the domain frequency image is multiply with high pass
    # filter then the center of domain frequency image(low Frequency pixels) becomes zero
    # ---------------------------------------------------------------
    # inverse fourier transform
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)  # absolute value

    return img_back


# with open-cv
def inverse_fourier_transform_method2(im):
    # fourier transform
    dft = cv2.dft(np.float32(im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # If you do not want to filter and only want the original image recovered
    # than comment the code for frequency domain filtering
    # ---------------------------------------------------------------
    # frequency domain filtering - ideal low pass filter
    rows, cols = im.shape
    r, c = rows // 2, cols // 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)  # shape = (rows, cols, 2) --> 2 channel - then use pic from opencv
    mask[r - 30:r + 30, c - 30:c + 30] = 1
    dft_shift = dft_shift * mask
    # ---------------------------------------------------------------
    # inverse fourier transform
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


# input image
image_name = input("please input image name(example : 1.JPG):")
gray_image = cv2.imread(image_name, 0)


# -----------------------------------------------------------------
# method1
# inverse fourier transform with numpy
# If you want to use this method, uncomment on the following line
# img_after_filtering = inverse_fourier_transform_method1(gray_image)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method2
# inverse fourier transform with open-cv
img_after_filtering = inverse_fourier_transform_method2(gray_image)
# -----------------------------------------------------------------

# plot
plt.figure(figsize=(19, 16))
plt.subplot(121), plt.imshow(gray_image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_after_filtering, cmap='gray')
plt.title('Image after frequency domain filtering'), plt.xticks([]), plt.yticks([])
plt.show()
