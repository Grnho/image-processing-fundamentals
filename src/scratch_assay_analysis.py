from matplotlib import pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu

import glob
time = 0
time_list = []
area_list = []
path = "/Users/grnho/Desktop/image-processing-fundamentals/data/scratch_assay/*.*"

for file in sorted(glob.glob(path)):
    img = io.imread(file)
    print(file)
    entropy_img = entropy(img, disk(10))

    #plt.imshow(entropy_img)
    thresh = threshold_otsu(entropy_img)
    print(thresh)#Best value for separating the two

    binary = entropy_img <= thresh
   # print(binary)#binary image of 0 and 1
    plt.imshow(binary, cmap = plt.cm.grey)
    scratch_area = np.sum(binary)
    print("Time:", time, "Number of True Pixels:", scratch_area)
    area_list.append(scratch_area)
    time_list.append(time)
    time +=1
print(time_list, area_list)
plt.figure(figsize=(8,5))
plt.plot(time_list, area_list, "bo-")

from scipy.stats import linregress
print(linregress(time_list, area_list))
slope, intercept, r_value, p_value, std = linregress(time_list, area_list)
print("y=",slope,"x+",intercept)
print("R\N{SUPERSCRIPT TWO} = ",r_value**2)
    
