import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------- CHANGE IMAGE NAME HERE ----------
img = cv2.imread(r"C:\Users\91986\Desktop\Landscape image.jpeg", 0)

# Safety check
if img is None:
    print("Error: Image not found. Check file name and path.")
    exit()

# --------- Sampling Function ----------
def sampling(image, factor):
    return image[::factor, ::factor]

# --------- Quantization Function ----------
def quantization(image, levels):
    step = 256 // levels
    quantized = (image // step) * step
    return quantized

# Apply different sampling rates
sample_2 = sampling(img, 2)
sample_4 = sampling(img, 4)

# Apply different quantization levels
quant_4 = quantization(img, 4)
quant_16 = quantization(img, 16)

# Display results
titles = ["Original", "Sampling x2", "Sampling x4", 
          "Quantization 4 levels", "Quantization 16 levels"]

images = [img, sample_2, sample_4, quant_4, quant_16]

plt.figure(figsize=(10,8))

for i in range(len(images)):
    plt.subplot(3,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()