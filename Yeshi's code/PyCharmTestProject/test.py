import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import os
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('Scrolls/1QIsa_C35.jpg')
imgCopy = img.copy()
# scaled_img = cv2.resize(img, (0, 0), img, 0.3, 0.3)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# contrast = cv2.addWeighted(gray, 2, np.zeros(gray.shape, gray.dtype), 0, 0)

# thresholding with gaussianblur
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
# threshBlur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 15)

# thresholding Gaussian - Binary
gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 15)

# thresholding Mean - Binary
mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 15)

# otsu thresholding
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#median blur
median = cv2.medianBlur(otsu, 5)

# Change the current directory
# to specified directory
os.chdir('Scrolls/Results')

# Saves image
print("Saving the median image was successful:", cv2.imwrite('median.png', median))
print("Saving the gauss image was successful:", cv2.imwrite('gauss.png', gauss))
print("Saving the mean image was successful:", cv2.imwrite('mean.png', mean))
print("Saving the otsu image was successful:", cv2.imwrite('otsu.png', otsu))


# Image to string using pytesseract, does not work well. Multiple issues:
# 1. Language pack does most likely not support the hebrew that is used in the scrolls.
# 2. The images are not "clean" enough.
# print(pytesseract.image_to_string(gray, lang="heb"))

# Detecting Characters and drawing red rectangles over them - gray image
hImg, wImg = gray.shape
boxes = pytesseract.image_to_boxes(gray, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    # print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(gray, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

# Detecting Characters and drawing red rectangles over them - otsu
hImg, wImg = otsu.shape
boxes = pytesseract.image_to_boxes(otsu, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    # print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(otsu, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

# Detecting Characters and drawing red rectangles over them - otsu
hImg, wImg = gauss.shape
boxes = pytesseract.image_to_boxes(gauss, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    # print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(gauss, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

# Detecting Characters and drawing red rectangles over them - otsu with median blur
hImg, wImg = median.shape
boxes = pytesseract.image_to_boxes(median, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(median, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)
    cv2.rectangle(imgCopy, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

#saves images of bounding boxes around letters
print("Saving the gray segment image was successful:", cv2.imwrite('graySeg.png', gray))
print("Saving the otsu segment image was successful:", cv2.imwrite('otsuSeg.png', otsu))
print("Saving the gauss segment image was successful:", cv2.imwrite('gaussSeg.png', gauss))
print("Saving the otsu with median blur segment image was successful:", cv2.imwrite('medianSeg.png', median))
print("Saving the img with otsu with median blur segment image was successful:", cv2.imwrite('imgCopySeg.png', imgCopy))


# cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
# cv2.imshow('Original', img)
#
# cv2.namedWindow('Median blur', cv2.WINDOW_KEEPRATIO)
# cv2.imshow('Median blur', gray)
#
#
# cv2.namedWindow('Thresh with blur', cv2.WINDOW_KEEPRATIO)
# cv2.imshow('Thresh with blur', thresh)
# cv2.waitKey(0)
