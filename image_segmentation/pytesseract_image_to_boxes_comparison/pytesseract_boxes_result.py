# OpenCV
import cv2
# For changing the directory so that we can save images in a different folder
import os

import pytesseract

# Necessary for running pytesseract
# Info on how to get it running: https://github.com/tesseract-ocr/tesseract/blob/main/README.md
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Reads image of scroll
img = cv2.imread('PLACEHOLDER')

# Grayscales image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# adaptive thresholding Gaussian - Binary
gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 15)

# adaptive thresholding Mean - Binary
mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 15)

# otsu thresholding - Binary
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#median blur with Otsu
median = cv2.medianBlur(otsu, 5)

# Change the current directory
# to specified directory
# for placing the results
os.chdir('PLACEHOLDER')

# x = Distance between the top left corner of the box to the left frame
# y = Distance between the top of the box to the bottom frame
# w = Distance between the right side of the box to the left frame
# h = Distance between the bottom of the box to the bottom frame

# Detecting Characters and drawing red rectangles over them - gray image
h_img, w_img = gray.shape
boxes = pytesseract.image_to_boxes(gray, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(gray, (x, h_img - y), (w, h_img - h), (0, 0, 255), 1)

# Detecting Characters and drawing red rectangles over them - otsu
h_img, w_img = otsu.shape
boxes = pytesseract.image_to_boxes(otsu, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(otsu, (x, h_img - y), (w, h_img - h), (0, 0, 255), 1)

# Detecting Characters and drawing red rectangles over them - gauss
h_img, w_img = gauss.shape
boxes = pytesseract.image_to_boxes(gauss, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(gauss, (x, h_img - y), (w, h_img - h), (0, 0, 255), 1)

# Detecting Characters and drawing red rectangles over them - otsu with median blur
h_img, w_img = median.shape
boxes = pytesseract.image_to_boxes(median, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(median, (x, h_img - y), (w, h_img - h), (0, 0, 255), 1)

#saves images of bounding boxes around letters
print("Saving the gray segment image was successful:", cv2.imwrite('gray_seg.png', gray))
print("Saving the otsu segment image was successful:", cv2.imwrite('otsu_seg.png', otsu))
print("Saving the gauss segment image was successful:", cv2.imwrite('gaussSeg.png', gauss))
print("Saving the otsu with median blur segment image was successful:", cv2.imwrite('otsu_median_seg.png', median))
