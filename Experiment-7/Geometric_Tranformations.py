import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Read Image
# -------------------------------
img = cv2.imread('C:/Users/91986/Desktop/Geometric_Image.jpeg')

if img is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB (for matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[:2]

# -------------------------------
# 1. Translation
# -------------------------------
tx, ty = 100, 50   # shift right and down
translation_matrix = np.float32([[1, 0, tx],
                                 [0, 1, ty]])

translated_img = cv2.warpAffine(img_rgb, translation_matrix, (cols, rows))

# -------------------------------
# 2. Scaling
# -------------------------------
scaled_img = cv2.resize(img_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# -------------------------------
# 3. Rotation
# -------------------------------
angle = 45
center = (cols // 2, rows // 2)

rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
rotated_img = cv2.warpAffine(img_rgb, rotation_matrix, (cols, rows))

# -------------------------------
# Display Results
# -------------------------------
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Translated Image")
plt.imshow(translated_img)
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Scaled Image")
plt.imshow(scaled_img)
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Rotated Image")
plt.imshow(rotated_img)
plt.axis('off')

plt.tight_layout()
plt.show()
