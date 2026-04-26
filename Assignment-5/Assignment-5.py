"""
Name: Prerna Kandpal
Roll No: XXXXX
Course: Image Processing & Computer Vision
Assignment Title: Intelligent Image Enhancement & Analysis System
Date:
"""

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

print("🧠 Intelligent Image Enhancement & Analysis System")
print("This system enhances, restores, segments, and analyzes images.\n")

# ------------------ CREATE OUTPUT FOLDER ------------------
os.makedirs("outputs", exist_ok=True)

# ------------------ TASK 2: IMAGE ACQUISITION ------------------
def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# ------------------ TASK 3: ADD NOISE ------------------
def add_gaussian_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy

def add_salt_pepper(image):
    noisy = image.copy()
    prob = 0.02
    rand = np.random.rand(*image.shape)

    noisy[rand < prob] = 0
    noisy[rand > 1 - prob] = 255

    return noisy

# ------------------ RESTORATION ------------------
def apply_filters(image):
    mean = cv2.blur(image, (5, 5))
    median = cv2.medianBlur(image, 5)
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    return mean, median, gaussian

# ------------------ ENHANCEMENT ------------------
def enhance_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0)
    enhanced = clahe.apply(image)
    return enhanced

# ------------------ TASK 4: SEGMENTATION ------------------
def segmentation(image):
    _, global_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return global_th, otsu

# ------------------ MORPHOLOGY ------------------
def morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(image, kernel, iterations=1)
    ero = cv2.erode(image, kernel, iterations=1)
    return dil, ero

# ------------------ TASK 5: FEATURES ------------------
def edges(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)

    canny = cv2.Canny(image, 100, 200)
    return sobel, canny

def contours(image, original):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = original.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return output

def orb_features(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return cv2.drawKeypoints(image, kp, None)

# ------------------ TASK 6: METRICS ------------------
def calculate_metrics(original, processed):
    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    s = ssim(original, processed)

    return mse, psnr, s

# ------------------ MAIN PIPELINE ------------------
def process_image(path, name):
    print(f"\n📊 Processing: {name}")

    original, gray = load_image(path)

    # Noise
    g_noise = add_gaussian_noise(gray)
    sp_noise = add_salt_pepper(gray)

    # Restoration
    mean, median, gaussian = apply_filters(sp_noise)

    # Enhancement
    enhanced = enhance_image(median)

    # Segmentation
    global_th, otsu = segmentation(enhanced)

    # Morphology
    dil, ero = morphology(otsu)

    # Features
    sobel, canny = edges(enhanced)
    cont = contours(canny, original)
    orb = orb_features(enhanced)

    # Metrics
    mse, psnr, s = calculate_metrics(gray, enhanced)

    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {s:.3f}")

    # Save outputs
    cv2.imwrite(f"outputs/{name}_original.png", original)
    cv2.imwrite(f"outputs/{name}_noisy.png", sp_noise)
    cv2.imwrite(f"outputs/{name}_restored.png", median)
    cv2.imwrite(f"outputs/{name}_enhanced.png", enhanced)
    cv2.imwrite(f"outputs/{name}_segmented.png", otsu)
    cv2.imwrite(f"outputs/{name}_features.png", orb)

    # Visualization
    titles = ['Original', 'Noisy', 'Restored', 'Enhanced', 'Segmented', 'Features']
    images = [gray, sp_noise, median, enhanced, otsu, orb]

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"outputs/{name}_comparison.png")
    plt.close()

    print("✅ Done!")


# ------------------ DRIVER ------------------
if __name__ == "__main__":
    images = [
        ("images/face.jpg", "face"),
        ("images/object.jpg", "object"),
        ("images/naturalScene.jpg", "naturalScene")
    ]

    for path, name in images:
        process_image(path, name)

    print("\n🎉 All Processing Completed Successfully!")