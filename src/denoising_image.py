from skimage import io, img_as_float
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from skimage import img_as_ubyte

image = img_as_float(io.imread("/Users/grnho/Desktop/image-processing-fundamentals/data/noisy_img.jpg"))

guassian_img = nd.gaussian_filter(image, sigma =3)
plt.imsave("/Users/grnho/Desktop/image-processing-fundamentals/data/noisy_gaussian.jpg", guassian_img)

median_img = nd.median_filter(image, size =3)
plt.imsave("/Users/grnho/Desktop/image-processing-fundamentals/data/median_gaussian.jpg",median_img)

sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))

patch_kw = dict(patch_size=5,
                patch_distance=3,
                multichannel=True)

denoise_img = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=False,
                               patch_size=5, patch_distance=3, channel_axis=-1)
"""
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
"""
denoise_img_as_8byte = img_as_ubyte(denoise_img)

plt.imshow(denoise_img)
#plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
plt.imsave("/Users/grnho/Desktop/image-processing-fundamentals/data/nlm_denoise.jpg",denoise_img_as_8byte)
