# PGM Image Processor in C

This project is a high-performance command-line tool developed in **C** for advanced image processing tasks on PGM (Portable Gray Map) files.

## 🛠 Key Features
* **Format Support:** Handles both ASCII (P2) and Binary (P5) PGM formats.
* **Edge Detection Suite:** Includes Sobel, Prewitt, and a complete 4-stage **Canny Edge Detector** (Gaussian Blur, Gradient Calculation, Non-Maximum Suppression, and Hysteresis Thresholding).
* **Texture Analysis:** Implements **Local Binary Pattern (LBP)** algorithm for feature extraction.
* **Image Manipulation:** Supports resizing (Nearest Neighbor zoom/shrink) and noise reduction filters (Average and Median).
* **Memory Management:** Efficiently handles dynamic 2D arrays and file I/O operations in C.

## 🚀 How to Run
1. Compile the code: `gcc image_processor.c -o processor -lm`
2. Run the application: `./processor`
3. Follow the on-screen menu to load an image and apply operations.
