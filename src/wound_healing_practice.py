from matplotlib import pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
url = "https://raw.githubusercontent.com/bnsreenu/python_for_microscopists/master/images/scratch_assay/Scratch0.jpg"
img = io.imread(url)
from skimage.filters import try_all_threshold

entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img, cmap = "grey")

fig, ax= try_all_threshold(entropy_img, figsize = (10,8), verbose = False)

fig, axes = plt.subplots(nrows = 2, ncols =2, sharex = True, sharey = True, figsize = (8,8))

ax = axes.ravel()

from skimage.filters import threshold_otsu
thresh = threshold_otsu(entropy_img)

binary = entropy_img <= thresh
ax[0].imshow(binary, cmap = plt.cm.grey)
ax[0].set_title("Binary")

print(
    "The percentage of bright pixels is:",
    np.mean(binary) * 100
)
