import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.rcParams['figure.figsize'] = (6.0, 6.0)
plt.rcParams['image.cmap'] = 'gray'
sig_org = cv2.imread('signature.jpg', cv2.IMREAD_COLOR)

# Crop the original image
sig = sig_org[700:2100, 450:3500, :]

# Convert to gray
sig_gray = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)

# Create an alpha mask using inverse thresholding
ret, alpha_mask = cv2.threshold(sig_gray, 170, 255, cv2.THRESH_BINARY_INV)

# We can enhance the color to look like blue ink
blue_mask = sig.copy()
blue_mask[:, :] = (255, 0, 0)

sig_color = cv2.addWeighted(sig, 1, blue_mask, 0.5, 0)

# Add the alpha mask as the 4th channel to the image

# split the color channels from the color image
b, g, r = cv2.split(sig_color)

new = [b, g, r, alpha_mask]

png = cv2.merge(new, 4)

cv2.imwrite('extracted_sig.png', png)
