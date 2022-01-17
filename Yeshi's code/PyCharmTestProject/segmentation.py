import cv2
import pytesseract
import os
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('TestImages/testIMG.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# otsu thresholding
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Change the current directory
# to specified directory
os.chdir('Scrolls/SegmentResults')

id = 0

test = otsu[(1536-1484):(1536-1456), 84:102]
print("Saving test Image", cv2.imwrite('test.png', test))

# Detecting Characters and drawing red rectangles over them - otsu
hImg, wImg = otsu.shape
print(otsu.shape)
boxes = pytesseract.image_to_boxes(otsu)
for b in boxes.splitlines():
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    crop = otsu[(hImg-h):(hImg-y), x:w]
    print(("Saving image number:" + str(id)), cv2.imwrite(str(id) + '.png', crop))
    id += 1

# # Detecting Characters and drawing red rectangles over them - otsu
# # Using PIL package
# hImg, wImg = otsu.shape
# boxes = pytesseract.image_to_boxes(otsu, lang="heb")
# for b in boxes.splitlines():
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     crop = otsu.crop((x,hImg-y,x+w,hImg-y-h))
#     print(("Saving image number:" + str(id)), cv2.imwrite(str(id) + '.png', crop))
#     id += 1

# Detecting Characters and drawing red rectangles over them - otsu
# hImg, wImg = otsu.shape
# print(otsu.shape)
# boxes = pytesseract.image_to_boxes(otsu)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     crop = cv2.rectangle(otsu, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)
#     print(("Saving image number:" + str(id)), cv2.imwrite(str(id) + '.png', crop))
#     id += 1

