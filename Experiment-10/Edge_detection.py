import cv2
import matplotlib.pyplot as plt

# -------------------------------
# Read Image
# -------------------------------
img = cv2.imread("C:/Users/91986/Desktop/inverse.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB (for display)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------
# Convert to Grayscale
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Apply Canny Edge Detection
# -------------------------------
edges = cv2.Canny(gray, 100, 200)

# -------------------------------
# Display Results
# -------------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Edge Detected Image")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()