import cv2
import numpy as np
import matplotlib.pyplot as plt



# Load image (grayscale)
img_path = r'C:\Users\91986\Desktop\InputImage.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded properly
if img is None:
    raise ValueError("Image not found. Check the file path.")

# FFT
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)

# Image dimensions
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2

# Create Low Pass Mask
mask_size = 30
low_pass_mask = np.zeros((rows, cols), np.uint8)
low_pass_mask[center_row-mask_size:center_row+mask_size,
              center_col-mask_size:center_col+mask_size] = 1

# High Pass Mask
high_pass_mask = 1 - low_pass_mask

# Apply filters
low_pass = np.fft.ifft2(np.fft.ifftshift(f_shift * low_pass_mask))
high_pass = np.fft.ifft2(np.fft.ifftshift(f_shift * high_pass_mask))

# Convert to magnitude
low_pass_img = np.abs(low_pass)
high_pass_img = np.abs(high_pass)

# Normalize for better display
low_pass_img = cv2.normalize(low_pass_img, None, 0, 255, cv2.NORM_MINMAX)
high_pass_img = cv2.normalize(high_pass_img, None, 0, 255, cv2.NORM_MINMAX)

# Plot results
plt.figure(figsize=(12, 5))

titles = ["Original Image", "Low Pass Filter", "High Pass Filter"]
images = [img, low_pass_img, high_pass_img]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()