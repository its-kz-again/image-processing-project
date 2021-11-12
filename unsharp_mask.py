import numpy as np
import matplotlib.pylab as plt
import cv2
from PIL import Image, ImageFilter
from skimage.filters import unsharp_mask


# unsharp mask with PIL library
def unsharp_mask_method1(name):
    imageObject = Image.open(name)
    sharpened1 = imageObject.filter(ImageFilter.SHARPEN)
    sharpened2 = sharpened1.filter(ImageFilter.SHARPEN)  # more sharper than sharpend1

    return imageObject, sharpened2


# unsharp mask with PIL library
def unsharp_mask_method2(im):
    image = Image.fromarray(im.astype('uint8'))
    sharpen = image.filter(ImageFilter.UnsharpMask(radius=3, percent=150))

    return sharpen


# unsharp mask with formula
def unsharp_mask_method3(im):
    im_blurred = cv2.GaussianBlur(im, (15, 15), 0)
    # sharpen = cv2.addWeighted(im, 1.0 + 3.0, im_blurred, -3.0, 0)  # for k = 3.0
    sharpen = cv2.addWeighted(im, 1.0 + 5.0, im_blurred, -5.0, 0)  # for k = 5.0
    # sharpen = cv2.addWeighted(im, 1.0 + 8.0, im_blurred, -8.0, 0)  # sharpen = im + 8.0 * (im - im_blurred) , k = 8.0

    return sharpen


# input image
image_name = input("please input image name(example : 1.JPG):")

# -----------------------------------------------------------------
# method1
# If you want to use this method, uncomment on the following line
# image, res = unsharp_mask_method1(image_name)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method2
# If you want to use this method, uncomment on the following lines
# image = plt.imread(image_name)
# res = unsharp_mask_method2(image)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method3
# unsharp mask with formula --> (k+1) * image  - k * blur(image)
# image = plt.imread(image_name) # for rgb image
image = cv2.imread(image_name, 0)
res = unsharp_mask_method3(image)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# method4
# unsharp mask for sharpened with skimage.filters.unsharp_mask
# If you want to use this method, uncomment on the following lines
# image = cv2.imread(image_name, 0)
# res = unsharp_mask(image, radius=1, amount=5)  # amount = k and radius is a parameter in gaussian filter
# -----------------------------------------------------------------


# plot image
for i in ([image, 'original image'], [res, 'sharpen image']):
    plt.figure(figsize=(30, 20))
    plt.imshow(i[0], cmap='gray')  # if you plot rgb image - should rewrite this code to --> plt.imshow(i[0])
    plt.title(i[1])
    plt.axis('off')
    plt.show()

