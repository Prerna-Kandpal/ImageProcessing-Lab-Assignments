import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Read Image
# -------------------------------
img = cv2.imread('C:/Users/91986/Desktop/color_segmentation.jpg')

if img is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB (for display)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------
# Convert RGB to HSV
# -------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -------------------------------
# Define Color Range (Example: Blue color)
# You can change values depending on your image
# -------------------------------
lower_color = np.array([100, 100, 100])
upper_color = np.array([140, 255, 255])

# -------------------------------
# Create Mask
# -------------------------------
mask = cv2.inRange(hsv, lower_color, upper_color)

# -------------------------------
# Apply Mask (Segmentation)
# -------------------------------
segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# -------------------------------
# Display Results
# -------------------------------
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2,2,2)
plt.title("HSV Image")
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Segmented Image")
plt.imshow(segmented)
plt.axis('off')

plt.tight_layout()
plt.show()

