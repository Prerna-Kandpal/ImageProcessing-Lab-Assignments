"""
Name: Prerna Kandpal
Roll No: XXXXX
Course: Image Processing & Computer Vision
Unit: Unit 1
Assignment Title: Feature-Based Traffic Monitoring System
Date:
"""

import cv2
import numpy as np
import os

print("🚦 Feature-Based Traffic Monitoring System Started")

# ------------------ CREATE OUTPUT FOLDERS ------------------
folders = [
    "outputs/edges",
    "outputs/contours",
    "outputs/features"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ------------------ TASK 1: EDGE DETECTION ------------------
def sobel_edge(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.absolute(sobel))

    return sobel


def canny_edge(image):
    edges = cv2.Canny(image, 100, 200)
    return edges


# ------------------ TASK 2: OBJECT REPRESENTATION ------------------
def detect_contours(image, original, name):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = original.copy()

    print(f"\n📦 Objects detected in {name}: {len(contours)}")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area > 500:  # filter noise
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            print(f"Object {i+1}: Area={area:.2f}, Perimeter={perimeter:.2f}")

    return output


# ------------------ TASK 3: FEATURE EXTRACTION ------------------
def extract_orb_features(image):
    orb = cv2.ORB_create(nfeatures=500)

    keypoints, descriptors = orb.detectAndCompute(image, None)

    output = cv2.drawKeypoints(image, keypoints, None)

    print(f"🔍 ORB Keypoints detected: {len(keypoints)}")

    return output


# ------------------ MAIN PROCESS FUNCTION ------------------
def process_image(path, name):
    print(f"\n📊 Processing Image: {name}")

    image_color = cv2.imread(path)
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    if image_color is None:
        print(f"❌ Error loading {path}")
        return

    # ---------- TASK 1: EDGE DETECTION ----------
    sobel = sobel_edge(gray)
    canny = canny_edge(gray)

    cv2.imwrite(f"outputs/edges/{name}_sobel.png", sobel)
    cv2.imwrite(f"outputs/edges/{name}_canny.png", canny)

    print("✅ Edge Detection Completed (Sobel vs Canny)")

    # ---------- TASK 2: CONTOURS ----------
    contour_img = detect_contours(canny, image_color, name)
    cv2.imwrite(f"outputs/contours/{name}_contours.png", contour_img)

    print("✅ Contour Detection Completed")

    # ---------- TASK 3: FEATURE EXTRACTION ----------
    orb_img = extract_orb_features(gray)
    cv2.imwrite(f"outputs/features/{name}_orb.png", orb_img)

    print("✅ Feature Extraction Completed")

    print("🎉 Processing Done for this Image!")


# ------------------ MAIN DRIVER ------------------
if __name__ == "__main__":
    print("\n📂 Make sure you have an 'images' folder with traffic images")

    images = [
        ("images/road1.jpg", "road1"),
        ("images/road2.jpg", "road2"),
        ("images/road3.jpg", "road3")
    ]

    for path, name in images:
        process_image(path, name)

    print("\n🚀 All Images Processed Successfully!")