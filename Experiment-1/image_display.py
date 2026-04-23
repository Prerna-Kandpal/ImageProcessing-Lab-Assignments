import cv2
import matplotlib.pyplot as plt

# Step 1: Load the image (Change file path if needed)
img = cv2.imread("C:/Users/91986/Desktop/Landscape2 image.png")

# Step 2: Check if image is loaded properly
if img is None:
    print("Error: Image not found!")
    exit()

# Step 3: Convert BGR to RGB (for correct color display)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 4: Display the image
plt.imshow(img_rgb)
plt.title("Acquired Image")
plt.axis('off')
plt.show()