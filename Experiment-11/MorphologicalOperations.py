import cv2
import numpy as np

# Step 1: Read image in grayscale
img = cv2.imread('binary.png', 0)

# Check image validity
if img is None:
    print("Error: Image not found")
    exit()

# Step 2: Define structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# Step 3: Apply Morphological Operations
dilation = cv2.dilate(img, kernel, iterations=1)
erosion = cv2.erode(img, kernel, iterations=1)

# Step 4: Display results
cv2.imshow("Original Binary Image", img)
cv2.imshow("Dilation", dilation)
cv2.imshow("Erosion", erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()