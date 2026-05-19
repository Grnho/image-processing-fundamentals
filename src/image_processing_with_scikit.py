from skimage import io, img_as_ubyte
from matplotlib import pyplot as plt
image = io.imread("/Users/grnho/Desktop/image-processing-fundamentals/data/test_image.jpg", as_gray = True)
from skimage.transform import rescale, resize, downscale_local_mean

rescaled_img = rescale(image, 1.0/4.0, anti_aliasing = True)
resized_img = resize(image, (200, 200))
downscaled_img = downscale_local_mean(image, (4,3))
#averageing each 4x3 pixel into 1

plt.imshow(downscaled_img)

from skimage.filters import roberts, sobel, scharr, prewitt

edge_roberts = roberts(image)
plt.imshow(edge_roberts, cmap = "grey")
edge_sobel = sobel(image)
edge_scharr = scharr(image)
edge_prewitt = prewitt(image)
fig, axes = plt.subplots(nrows = 2, ncols =2, sharex = True, sharey = True, figsize = (8,8))

ax = axes.ravel()

ax[0].imshow(image, cmap = plt.cm.grey)
ax[0].set_title("Original")

ax[1].imshow(edge_sobel, cmap = plt.cm.grey)
ax[1].set_title("edge_sobel")

ax[2].imshow(edge_scharr, cmap = plt.cm.grey)
ax[2].set_title("edge_scharr")

ax[3].imshow(edge_prewitt, cmap = plt.cm.grey)
ax[3].set_title("edge_prewitt")

for a in ax:
    a.axis("off")
    
plt.tight_layout
plt.show()

from skimage.feature import canny

edge_canny = canny(image, sigma = 1)
plt.imshow(edge_canny)

from skimage import restoration
import numpy as np
psf = np.ones((3,3))/9

deconvolved, _ = restoration.unsupervised_wiener(image, psf)
io.imsave(
    "/Users/grnho/Desktop/image-processing-fundamentals/data/deconvolved.jpg",
    img_as_ubyte(deconvolved)
)
