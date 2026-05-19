from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread("../data/neuron.jpg", 0)
edges = cv2.Canny(img, 100, 200)
"""
gradient > 200
definitely edge

gradient < 100
definitely not edge

100~200
maybe edge
"""
plt.figure(figsize = (4,3))
plt.title("Original Image")
plt.imshow(img, cmap = "grey")

plt.figure(figsize = (4,3))
plt.title("Edge Detection")
plt.imshow(edges, cmap = plt.cm.grey)
