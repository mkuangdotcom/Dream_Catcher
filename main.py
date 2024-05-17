# Image Processing
import cv2              # For computer vision and image processing
import numpy as np      # Support for large, multi-dimensional arrays and matrices of numerical data and collection of mathematical functions

cap = cv2.VideoCapture(0)  # use the first camera device it finds.

while True:
    ret, frame = cap.read()  # Capture frame-by-frame, ret indicates if the frame was captured successfully (boolean), frame NumPy array containing the pixel data for the frame.