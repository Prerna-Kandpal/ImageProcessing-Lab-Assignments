"""
------------------------------------------------------------
Name: Prerna kandpal
Roll No: 2301010251
Course: Image Processing
Unit: Noise Removal & Restoration
Assignment Title: Image Restoration using Spatial Filtering
Date: 2026-04-24
------------------------------------------------------------
"""

import cv2
import numpy as np
import os
from math import log10, sqrt

# ===========================
# CONFIG (Dummy Paths)
# ===========================

INPUT_FOLDER = "inputs"
OUTPUT_FOLDER = "outputs"

# Dummy surveillance-style filenames
IMAGE_PATHS = [
    "inputs/street.jpg",
    "inputs/parking.jpg",
    "inputs/corridor.jpg"
]

# ===========================
# Utility Functions
# ===========================

def create_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def generate_dummy_image(path):
    """
    Creates a synthetic grayscale image if file doesn't exist.
    """
    print(f"[INFO] Generating dummy image: {path}")
    
    img = np.zeros((256, 256), dtype=np.uint8)

    # Draw shapes to simulate objects
    cv2.rectangle(img, (50, 50), (200, 200), 150, -1)
    cv2.circle(img, (128, 128), 40, 220, -1)
    cv2.line(img, (0, 0), (255, 255), 100, 2)

    cv2.imwrite(path, img)


def load_image(path):
    if not os.path.exists(path):
        generate_dummy_image(path)

    print(f"\n[INFO] Loading image: {path}")
    img = cv2.imread(path)

    if img is None:
        raise Exception(f"Failed to load {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


def save_image(filename, image):
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), image)


def mse(original, restored):
    return np.mean((original - restored) ** 2)


def psnr(original, restored):
    error = mse(original, restored)
    if error == 0:
        return 100
    return 20 * log10(255.0 / sqrt(error))


# ===========================
# Noise Functions
# ===========================

def add_gaussian_noise(image):
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, image.shape)

    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image):
    noisy = np.copy(image)
    prob = 0.02

    salt = np.random.rand(*image.shape) < prob
    pepper = np.random.rand(*image.shape) < prob

    noisy[salt] = 255
    noisy[pepper] = 0

    return noisy


# ===========================
# Filters
# ===========================

def apply_filters(image):
    mean = cv2.blur(image, (5, 5))
    median = cv2.medianBlur(image, 5)
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    return mean, median, gaussian


# ===========================
# Evaluation
# ===========================

def evaluate(original, images):
    names = ["Mean", "Median", "Gaussian"]
    results = []

    for name, img in zip(names, images):
        m = mse(original, img)
        p = psnr(original, img)
        results.append((name, m, p))

    return results


def print_results(results, title):
    print(f"\n===== {title} =====")
    for name, m, p in results:
        print(f"{name:10s} | MSE: {m:8.2f} | PSNR: {p:6.2f} dB")


def analyze(results, noise_type):
    best = max(results, key=lambda x: x[2])

    print(f"\n[ANALYSIS] {noise_type}")
    print(f"Best Filter: {best[0]}")

    if noise_type == "Gaussian":
        print("Reason: Gaussian filter best matches Gaussian noise distribution.")
    else:
        print("Reason: Median filter removes impulse noise effectively.")


# ===========================
# Main Processing Per Image
# ===========================

def process_image(path, index):
    original = load_image(path)

    save_image(f"{index}_original.png", original)

    # Add noise
    g_noise = add_gaussian_noise(original)
    sp_noise = add_salt_pepper_noise(original)

    save_image(f"{index}_gaussian_noise.png", g_noise)
    save_image(f"{index}_saltpepper_noise.png", sp_noise)

    # Apply filters
    g_filters = apply_filters(g_noise)
    sp_filters = apply_filters(sp_noise)

    # Save filtered images
    filter_names = ["mean", "median", "gaussian"]

    for name, img in zip(filter_names, g_filters):
        save_image(f"{index}_g_{name}.png", img)

    for name, img in zip(filter_names, sp_filters):
        save_image(f"{index}_sp_{name}.png", img)

    # Evaluate
    g_results = evaluate(original, g_filters)
    sp_results = evaluate(original, sp_filters)

    print_results(g_results, f"Image {index} - Gaussian Noise")
    print_results(sp_results, f"Image {index} - Salt & Pepper Noise")

    analyze(g_results, "Gaussian")
    analyze(sp_results, "Salt & Pepper")


# ===========================
# MAIN
# ===========================

def main():
    create_folders()

    print("\n===== Image Restoration Assignment =====")

    for i, path in enumerate(IMAGE_PATHS, start=1):
        process_image(path, i)

    print("\n[INFO] Done! Check 'outputs/' folder.")


if __name__ == "__main__":
    main()