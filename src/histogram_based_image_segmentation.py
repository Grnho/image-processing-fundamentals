from skimage import io, img_as_float
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from skimage import img_as_ubyte

image = img_as_float(
    io.imread("/Users/grnho/Desktop/image-processing-fundamentals/data/BSE_Google.jpg")
)

print(image.shape)
print(image.ndim)
print(image.dtype)
sigma_est = np.mean(estimate_sigma(image, channel_axis= None))

denoise_img = denoise_nl_means(
    image,
    h=1.15 * sigma_est,
    fast_mode=False,
    patch_size=5,
    patch_distance=3,
    channel_axis= None,
)

ubyte_img = img_as_ubyte(denoise_img)
plt.imshow(ubyte_img, cmap="gray")

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    sharex=True,
    sharey=True,
    figsize=(8,8)
)

ax = axes.ravel()

ax[0].hist(
    ubyte_img.flat,
    bins=100,
    range=(0,80)
)

plt.show()

segm1 = (ubyte_img<=55)
segm2 = (ubyte_img>55)&(ubyte_img<=110)
segm3 = (ubyte_img>110)&(ubyte_img<=210)
segm4 = (ubyte_img>210)

all_segments = np.zeros((ubyte_img.shape[0],ubyte_img.shape[1], 3))

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)
all_segments[segm3] = (0,0,1)
all_segments[segm4] = (1,1,0)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()

ax[0].hist(ubyte_img.flat, bins=100, range=(0, 80))
ax[0].set_title("Histogram")

ax[1].imshow(ubyte_img, cmap="gray")
ax[1].set_title("Denoised image")
ax[1].axis("off")

ax[2].imshow(all_segments)
ax[2].set_title("All segments")
ax[2].axis("off")

segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))
segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))
segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))
segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))

all_segments_cleaned = np.zeros((ubyte_img.shape[0],ubyte_img.shape[1], 3))
all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)

ax[3].imshow(all_segments_cleaned)
ax[3].set_title("All Segments Cleaned")
plt.tight_layout()
plt.show()
plt.imsave("/Users/grnho/Desktop/image-processing-fundamentals/data/cleaned_segmentation.jpg", all_segments_cleaned)
