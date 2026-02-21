## Lab Assignment 1
# 🖼️ Sampling and Quantization of a Digital Image

## 📌 Overview

This project demonstrates the concepts of **Sampling** and **Quantization** in Digital Image Processing using Python.  
The program visualizes how image quality changes when spatial resolution and intensity levels are reduced.

---

## 🎯 Aim

To develop a Python program to perform sampling and quantization on a given image and visualize the effect of different sampling rates and quantization levels on image quality.

---

## 🛠️ Tools & Technologies Used

- **Programming Language:** Python  
- **Libraries:**  
  - OpenCV  
  - NumPy  
  - Matplotlib  
- **IDE:** VS Code  

---

## 📖 Theory

### 🔹 Sampling

Sampling refers to reducing the spatial resolution of an image by decreasing the number of pixels.

- Higher sampling factor → Lower resolution  
- Image becomes blurred or pixelated  
- Fine details are lost  

### 🔹 Quantization

Quantization reduces the number of intensity (brightness) levels in an image.

- Lower quantization levels → Fewer shades  
- Image appears poster-like  
- Banding effect becomes visible  

---

## ⚙️ Procedure

1. Install required libraries:
   ```bash
   pip install opencv-python
   pip install numpy
   pip install matplotlib
   ```

2. Load a colored image.
3. Convert the image from BGR to RGB format.
4. Apply sampling with factors 2 and 4.
5. Apply quantization with 4 and 16 intensity levels.
6. Display the original and processed images for comparison.

---

## 💻 Python Implementation
- sampling_quantization.py
---

## 📊 Output

The program generates and displays:

1. Original Image  
2. Sampled Image (Factor 2)  
3. Sampled Image (Factor 4)  
4. Quantized Image (4 Intensity Levels)  
5. Quantized Image (16 Intensity Levels)  

Each output clearly shows how image quality changes with reduced spatial and intensity resolution.

---

## 🔍 Observations

- As the sampling factor increases, spatial resolution decreases.
- Sampling causes the image to appear blurred or pixelated.
- As quantization levels decrease, intensity transitions become less smooth.
- Lower quantization levels create visible banding effects.
- The original image retains maximum clarity and detail.

---

## ✅ Conclusion

Sampling reduces the spatial resolution of an image, while quantization reduces its intensity resolution.  
Both techniques demonstrate how reducing resolution impacts overall image quality.  
This experiment helps in understanding fundamental concepts used in image compression and digital image processing systems.

---

## 🚀 Applications

- Image Compression  
- Multimedia Systems  
- Medical Image Processing  
- Satellite Image Processing  
- Digital Photography  

---

## 👩‍💻 Author

**Prerna Kandpal**  
B.Tech CSE - D  
2301010251
