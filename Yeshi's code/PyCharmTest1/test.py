import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('Scrolls/havardscroll.png')
scaled_img = cv2.resize(img, (0, 0), img, 0.8, 0.8)
# histr = cv2.calcHist([scaled_img], [0], None, [256], [0,256])

gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
# contrast = cv2.addWeighted(gray, 2, np.zeros(gray.shape, gray.dtype), 0, 0)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 15)

# kernel = np.ones((2, 2), np.uint8)
# erosion = cv2.erode(contrast, kernel, iterations=1)
# dilation = cv2.dilate(erosion, kernel, iterations=1)

# kernel = np.ones((4,4),np.uint8)
# morph = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, kernel)

# # Detecting Characters
# hImg, wImg = img.shape
# boxes = pytesseract.image_to_boxes(img, lang="heb")
# for b in boxes.splitlines():
#     b = b.split(' ')
#     # print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

# plt.plot(histr)
# plt.show()
cv2.imshow('Original', scaled_img)
cv2.imshow('Gray', gray)
# cv2.imshow('Contrast', contrast)
cv2.imshow('Thresh', thresh)
cv2.waitKey(0)
