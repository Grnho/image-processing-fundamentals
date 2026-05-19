from skimage import io, img_as_ubyte
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt

img = img_as_ubyte(io.imread("/Users/grnho/Desktop/image-processing-fundamentals/data/monkey.jpg", as_gray = True))
print(img.shape, img.dtype)

print(img)

mean_grey = img.mean()
max_grey = img.max()
min_grey = img.min()

print("Min, Max, and Mean are:", min_grey, max_grey, mean_grey)

flippedLR = np.fliplr(img)
flippedUD = np.flipud(img)#up down


plt.subplot(2,1,1)
plt.imshow(img, cmap = "Greys")
plt.subplot(2,2,3)
plt.imshow(flippedLR, cmap = "Blues")

plt.subplot(2, 2, 4)
plt.imshow(flippedUD, cmap = "hsv")

uniformed_filter = ndimage.uniform_filter(img, size = 3)
plt.subplot(3,3,3)
plt.imshow(uniformed_filter)

guassian_img = ndimage.gaussian_filter(img, sigma = 11)
plt.subplot(3,3,1)
plt.imshow(guassian_img)

plt.figure()
sobel_img = ndimage.sobel(img, axis=0)
plt.imshow(sobel_img)

