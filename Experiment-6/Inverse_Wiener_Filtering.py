import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener

# -------------------------------
# Load Image (Grayscale)
# -------------------------------
img = cv2.imread("C:/Users/91986/Desktop/contraststreching.jpeg", 0)

if img is None:
    print("Error: Image not found!")
    exit()

# -------------------------------
# Add Periodic Noise
# -------------------------------
rows, cols = img.shape

x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

# Sinusoidal noise
noise = 30 * np.sin(2 * np.pi * X / 20)

# Add noise
noisy_img = img + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# -------------------------------
# Inverse Filtering
# -------------------------------
# Fourier Transform
f = np.fft.fft2(noisy_img)
fshift = np.fft.fftshift(f)

# Assume degradation function
H = np.ones_like(fshift)

# Avoid division by zero
H[H == 0] = 0.01

# Apply inverse filter
inverse_filter = fshift / H

# Inverse FFT
f_ishift = np.fft.ifftshift(inverse_filter)
img_inverse = np.fft.ifft2(f_ishift)
img_inverse = np.abs(img_inverse)

# Normalize result
img_inverse = cv2.normalize(img_inverse, None, 0, 255, cv2.NORM_MINMAX)
img_inverse = img_inverse.astype(np.uint8)

# -------------------------------
# Wiener Filtering
# -------------------------------
img_wiener = wiener(noisy_img, (5, 5))
img_wiener = np.clip(img_wiener, 0, 255).astype(np.uint8)

# -------------------------------
# Display Results
# -------------------------------
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Noisy Image")
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Inverse Filter Result")
plt.imshow(img_inverse, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Wiener Filter Result")
plt.imshow(img_wiener, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# Save Output Images (optional)
# -------------------------------
cv2.imwrite('noisy_image.jpg', noisy_img)
cv2.imwrite('inverse_filtered.jpg', img_inverse)
cv2.imwrite('wiener_filtered.jpg', img_wiener)

print("Processing completed. Images saved.")