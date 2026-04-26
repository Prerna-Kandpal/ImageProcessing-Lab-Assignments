"""
Name: Prerna Kandpal
Roll No: 2301010251
Course: Image Processing & Computer Vision
Unit: Unit 1
Assignment Title: Medical Image Compression & Segmentation System
"""

import cv2
import numpy as np
import os

print("🩺 Medical Image Compression & Segmentation System Started")

# ------------------ CREATE OUTPUT FOLDERS ------------------
folders = [
    "outputs/original",
    "outputs/segmented",
    "outputs/morphology",
    "outputs/compression"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ------------------ TASK 1: RLE COMPRESSION ------------------
def rle_encode(image):
    pixels = image.flatten()
    encoding = []

    prev_pixel = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoding.append((int(prev_pixel), count))
            prev_pixel = pixel
            count = 1

    encoding.append((int(prev_pixel), count))
    return encoding


def compression_stats(original, encoded):
    original_size = original.size
    compressed_size = len(encoded) * 2  # storing (value, count)

    ratio = original_size / compressed_size
    savings = (1 - (compressed_size / original_size)) * 100

    return ratio, savings


# ------------------ TASK 2: SEGMENTATION ------------------
def global_threshold(image, thresh=127):
    _, result = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return result


def otsu_threshold(image):
    print("🧠 Using Otsu's method for automatic threshold detection")
    _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


# ------------------ TASK 3: MORPHOLOGY ------------------
def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)

    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(image, kernel, iterations=1)

    return dilation, erosion


# ------------------ MAIN PROCESS FUNCTION ------------------
def process_image(path, name):
    print(f"\n📊 Processing Image: {name}")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ Error: Could not load {path}")
        return

    # Save original
    cv2.imwrite(f"outputs/original/{name}.png", image)

    # ---------- TASK 1: COMPRESSION ----------
    encoded = rle_encode(image)
    ratio, savings = compression_stats(image, encoded)

    print(f"📦 Compression Ratio: {ratio:.2f}")
    print(f"💾 Storage Savings: {savings:.2f}%")

    # Save RLE data
    with open(f"outputs/compression/{name}_rle.txt", "w") as f:
        f.write(str(encoded))

    # ---------- TASK 2: SEGMENTATION ----------
    global_seg = global_threshold(image)
    otsu_seg = otsu_threshold(image)

    cv2.imwrite(f"outputs/segmented/{name}_global.png", global_seg)
    cv2.imwrite(f"outputs/segmented/{name}_otsu.png", otsu_seg)

    # ---------- TASK 3: MORPHOLOGY ----------
    dil, ero = apply_morphology(otsu_seg)

    cv2.imwrite(f"outputs/morphology/{name}_dilation.png", dil)
    cv2.imwrite(f"outputs/morphology/{name}_erosion.png", ero)

    print("✅ Processing Completed Successfully!")


# ------------------ MAIN DRIVER ------------------
if __name__ == "__main__":
    print("\n📂 Make sure you have an 'images' folder with medical images")

    images = [
        ("images/xray.png", "xray"),
        ("images/mri.png", "mri"),
        ("images/ct.png", "ct")
    ]

    for path, name in images:
        process_image(path, name)

    print("\n🎉 All Images Processed Successfully!")