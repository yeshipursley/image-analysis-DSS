import cv2
import pytesseract
import os
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('Scrolls/1QIsa_C35.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# otsu thresholding
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Change the current directory
# to specified directory
os.chdir('Scrolls/ScrollSegmentResults')

id = 0

# Detecting Characters and drawing red rectangles over them - otsu
hImg, wImg = otsu.shape
print(otsu.shape)
boxes = pytesseract.image_to_boxes(otsu, lang="heb")
for b in boxes.splitlines():
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    crop = otsu[(hImg-h):(hImg-y), x:w]
    print(("Saving image number:" + str(id)), cv2.imwrite(str(id) + '.png', crop))
    id += 1