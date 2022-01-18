# OpenCV
import cv2

import pytesseract

# For changing the directory so that we can save images in a different folder
import os

# Necessary for running pytesseract
# Info on how to get it running: https://github.com/tesseract-ocr/tesseract/blob/main/README.md
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Reads image of scroll
img = cv2.imread('PLACEHOLDER')

# Grayscales image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# otsu thresholding
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Change the current directory
# to specified directory
# ADD desired folder for segmented letters/words
os.chdir('PLACEHOLDER')

#Incremented ID for cropped images
id = 0

## Crops the images around the letters/words
# Saves the hight and widht of the images
hImg, wImg = otsu.shape

# Makes a box around each letter/word on the scroll
boxes = pytesseract.image_to_boxes(otsu, lang="heb")

# For each box
for b in boxes.splitlines():
    # Save the coordinates of the box:
        # x = Distance between the top left corner of the box to the left frame
        # y = Distance between the top of the box to the bottom frame
        # w = Distance between the right side of the box to the left frame
        # h = Distance between the bottom of the box to the bottom frame
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

    # Crop the image so that we only get the letter/word
    # Structure image[rows, col]
    crop = otsu[(hImg-h):(hImg-y), x:w]

    # Saves the cropped images into the spesified folder written above.
    # Each image ID is incremented
    print(("Saving image number:" + str(id)), cv2.imwrite(str(id) + '.png', crop))
    id += 1