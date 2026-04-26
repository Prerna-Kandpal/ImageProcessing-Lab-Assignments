"""
Name: Prerna Kandpal
Roll No: 2301010251
Course: Image Processing & Computer Vision
Unit: Unit 1
Assignment Title: Smart Document Scanner & Quality Analysis System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("📄 Welcome to Smart Document Scanner & Quality Analysis System")

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ----------- TASK 2: IMAGE ACQUISITION -----------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# ----------- TASK 3: SAMPLING -----------
def sampling(gray):
    sizes = [512, 256, 128]
    sampled_images = []

    for size in sizes:
        down = cv2.resize(gray, (size, size))
        up = cv2.resize(down, (512, 512))  # upscale for display
        sampled_images.append(up)

    return sampled_images

# ----------- TASK 4: QUANTIZATION -----------
def quantize(gray, levels):
    step = 256 // levels
    quantized = (gray // step) * step
    return quantized

# ----------- TASK 5: DISPLAY & SAVE -----------
def display_results(original, gray, sampled, quantized, name):
    plt.figure(figsize=(12, 8))

    # Original & grayscale
    plt.subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    # Sampling
    titles = ["512x512", "256x256", "128x128"]
    for i in range(3):
        plt.subplot(2, 4, i+3)
        plt.title(titles[i])
        plt.imshow(sampled[i], cmap='gray')
        plt.axis('off')

    # Quantization
    q_titles = ["8-bit", "4-bit", "2-bit"]
    for i in range(3):
        plt.subplot(2, 4, i+6)
        plt.title(q_titles[i])
        plt.imshow(quantized[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"outputs/comparison_{name}.png")
    plt.show()

# ----------- MAIN FUNCTION -----------
def process_image(path, name):
    original, gray = load_image(path)

    # Sampling
    sampled = sampling(gray)

    # Quantization
    q8 = quantize(gray, 256)
    q4 = quantize(gray, 16)
    q2 = quantize(gray, 4)
    quantized = [q8, q4, q2]

    # Save outputs
    cv2.imwrite(f"outputs/{name}_gray.png", gray)
    cv2.imwrite(f"outputs/{name}_256.png", sampled[0])
    cv2.imwrite(f"outputs/{name}_128.png", sampled[2])
    cv2.imwrite(f"outputs/{name}_q4.png", q4)
    cv2.imwrite(f"outputs/{name}_q2.png", q2)

    # Display
    display_results(original, gray, sampled, quantized, name)


# ----------- RUN FOR MULTIPLE IMAGES -----------
images = [
    ("inputs/notes.jpg", "notes"),
    ("inputs/paper.jpg", "paper"),
    ("inputs/bookPage.jpg", "bookPage")
]

for path, name in images:
    print(f"Processing {name}...")
    process_image(path, name)

print("✅ Processing Completed! Check outputs folder.")