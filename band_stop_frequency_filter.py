import numpy as np
import matplotlib.pylab as plt
import cv2
import random


# band stop filter
def createBSFilter(shape, center, bandCenter, bandWidth, lpType=2, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter_matrix = np.zeros(shape, np.float32)

    if lpType == 0:  # Ideal band stop filter
        lpFilter = np.copy(d)
        lpFilter[:, :] = 0
        lpFilter[d > (bandCenter + bandWidth / 2)] = 1
        lpFilter[d < (bandCenter - bandWidth / 2)] = 1

    elif lpType == 1:  # Butterworth band stop filter
        lpFilter = 1.0 / (1 + np.power(d * bandWidth / (abs(pow(d, 2) - pow(bandCenter, 2)) + 1), 2 * n))  # plus 1 (because divide by zero error)

    elif lpType == 2:  # Gaussian band stop filter
        lpFilter = 1 - np.exp(-pow((pow(d, 2) - pow(bandCenter, 2)) / ((d * bandWidth) + 1), 2))  # plus 1 (because divide by zero error)

    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    return lpFilter_matrix


# with open-cv
def stdFftImage(img_gray, rows, cols):
    fimg = np.copy(img_gray)
    fimg = fimg.astype(np.float32)  # Notice the type conversion here

    # 1.Image matrix times(-1)^(r+c), Centralization --> for shift dc component
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2:
                fimg[r][c] = -1 * img_gray[r][c]
    img_fft = fftImage(fimg, rows, cols)

    return img_fft


# fourier transform with open-cv
def fftImage(img_gray, rows, cols):
    rPadded = cv2.getOptimalDFTSize(rows)
    cPadded = cv2.getOptimalDFTSize(cols)
    imgPadded = np.zeros((rPadded, cPadded), dtype=np.float32)
    imgPadded[:rows, :cols] = img_gray
    img_fft = cv2.dft(imgPadded, flags=cv2.DFT_COMPLEX_OUTPUT)

    return img_fft


# calculate gray spectrum
def graySpectrum(fft_img):
    real = np.power(fft_img[:, :, 0], 2.0)
    imaginary = np.power(fft_img[:, :, 1], 2.0)
    amplitude = np.sqrt(real + imaginary)
    spectrum = np.log(amplitude + 1.0)
    spectrum = cv2.normalize(spectrum, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    spectrum *= 255

    return amplitude, spectrum


# input image
image_name = input("please input image name(example : 1.JPG):")
gray_image = cv2.imread(image_name, 0)

rows, cols = gray_image.shape[:2]
img_fft = stdFftImage(gray_image, rows, cols)  # fourier transform of gray_image
amplitude, _ = graySpectrum(img_fft)
minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)  # need to maxloc - maxloc is center of image - format = (y, x)
# --------------------------------------------------------
max_radius = np.sqrt(pow(rows, 2) + pow(cols, 2)) / 4
BW = random.randint(0, int(max_radius))
BC = random.randint(0, int(max_radius))
BC = 60
BW = 60
# --------------------------------------------------------
# create band stop filters
BsFilter_1 = createBSFilter(img_fft.shape, maxLoc, BC, BW, 0)  # ideal band-stop filter
BsFilter_2 = createBSFilter(img_fft.shape, maxLoc, BC, BW, 1)  # Butterworth band-stop filter
BsFilter_3 = createBSFilter(img_fft.shape, maxLoc, BC, BW, 2)  # Gaussian band-stop pass filter
# --------------------------------------------------------
img_filter1 = BsFilter_1 * img_fft
img_filter2 = BsFilter_2 * img_fft
img_filter3 = BsFilter_3 * img_fft

_, gray_spectrum1 = graySpectrum(img_filter1)  # Observe the change of the filter
_, gray_spectrum2 = graySpectrum(img_filter2)
_, gray_spectrum3 = graySpectrum(img_filter3)

# idft
img_ift1 = cv2.dft(img_filter1, flags=cv2.DFT_INVERSE+cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE)
img_ift2 = cv2.dft(img_filter2, flags=cv2.DFT_INVERSE+cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE)
img_ift3 = cv2.dft(img_filter3, flags=cv2.DFT_INVERSE+cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE)
ori_img1 = np.copy(img_ift1[:rows, :cols])
ori_img2 = np.copy(img_ift2[:rows, :cols])
ori_img3 = np.copy(img_ift3[:rows, :cols])

# decentralize --> for reverse shift dc component
for r in range(rows):
    for c in range(cols):
        if (r + c) % 2:
            ori_img1[r][c] = -1 * ori_img1[r][c]
            ori_img2[r][c] = -1 * ori_img2[r][c]
            ori_img3[r][c] = -1 * ori_img3[r][c]

        # Truncate high and low values
        if ori_img1[r][c] < 0:
            ori_img1[r][c] = 0
        if ori_img1[r][c] > 255:
            ori_img1[r][c] = 255

        if ori_img2[r][c] < 0:
            ori_img2[r][c] = 0
        if ori_img2[r][c] > 255:
            ori_img2[r][c] = 255

        if ori_img3[r][c] < 0:
            ori_img3[r][c] = 0
        if ori_img3[r][c] > 255:
            ori_img3[r][c] = 255

# original filtered images
ori_img1 = ori_img1.astype(np.uint8)
ori_img2 = ori_img2.astype(np.uint8)
ori_img3 = ori_img3.astype(np.uint8)

# plot
# plot filters and gray spectrum
for i in ([[BsFilter_1, 'ideal', gray_spectrum1], [BsFilter_2, 'Butterworth', gray_spectrum2
                                                   ], [BsFilter_3, 'Gaussian', gray_spectrum3]]):
    plt.figure(figsize=(19, 16))
    plt.subplot(121), plt.imshow(cv2.magnitude(i[0][:, :, 0], i[0][:, :, 1]), cmap='gray')
    plt.title('{} stop-band filter'.format(i[1])), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(i[2], cmap='gray')
    plt.title('gray spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

# plt original image
plt.figure(figsize=(20, 16))
plt.imshow(gray_image, cmap='gray')
plt.title('original gray scale image')
plt.axis('off')
plt.show()
# plt original image with filters
plt.figure(figsize=(20, 16))
plt.subplot(131), plt.imshow(ori_img1, cmap='gray'), plt.title('image with ideal stop-band filter')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(ori_img2, cmap='gray'), plt.title('image with Butterworth stop-band filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(ori_img3, cmap='gray'), plt.title('image with Gaussian  stop-band filter')
plt.xticks([]), plt.yticks([])
plt.show()
