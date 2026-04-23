## Experiment 1
#  Image Acquisition and Display Using Python
This experiment demonstrates how to acquire (read) and display an image using Python and OpenCV.

---

## 🎯 Aim

To implement a program to acquire and display an image using Python.

---

## 🛠️ Tools & Technologies Used

- Programming Language: Python  
- Libraries:
  - OpenCV
  - NumPy
  - Matplotlib
- IDE: VS Code

---

## 📖 Theory

Image acquisition is the process of capturing or reading an image from a source such as:

- Camera
- Scanner
- Image file

In this experiment, we acquire an image from the system storage and display it using Python.

OpenCV provides the function `cv2.imread()` to read an image and `matplotlib` to display it.

---

## ⚙️ Procedure

1. Install required libraries:
   ```bash
   pip install opencv-python
   pip install numpy
   pip install matplotlib
   ```

2. Import required libraries.
3. Load the image using `cv2.imread()`.
4. Convert BGR image to RGB format.
5. Display the image using matplotlib.

---

## 💻 Python Implementation
- image_display.py

---

## 📊 Output

- The program successfully reads the image from the system.
- The image is displayed in a new window.
- If the image path is incorrect, an error message is shown.

---

## 🔍 Observations

- Image is loaded from storage successfully.
- BGR to RGB conversion is necessary for correct color display.
- Proper file path is important to avoid errors.

---

## ✅ Conclusion

The experiment successfully demonstrates image acquisition and display using Python and OpenCV. This forms the basic step in digital image processing applications.

---

## 🚀 Applications

- Image Processing Systems  
- Computer Vision  
- Face Recognition  
- Medical Imaging  
- Surveillance Systems  

---
