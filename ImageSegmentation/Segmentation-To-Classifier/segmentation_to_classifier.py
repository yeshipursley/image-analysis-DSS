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

# Denoises the otsu image
deNoiseOtsu = cv2.fastNlMeansDenoising(otsu, h=60.0, templateWindowSize=7, searchWindowSize=21)

# Change the current directory
# to specified directory
# ADD desired folder for segmented letters/words
os.chdir('PLACEHOLDER')

# Incremented ID for cropped images
id = 0

## Crops the images around the letters/words
# Saves the height and width of the images
hImg, wImg = deNoiseOtsu.shape

# array for the segmented letters
segmentedLetters = []

# Makes a box around each letter/word on the scroll
boxes = pytesseract.image_to_boxes(deNoiseOtsu, lang="heb")

# For each box
for b in boxes.splitlines():
    # Splits the values of the box into an array
    b = b.split(' ')
    # Save the coordinates of the box:
    # x = Distance between the top left corner of the box to the left frame
    # y = Distance between the top of the box to the bottom frame
    # w = Distance between the right side of the box to the left frame
    # h = Distance between the bottom of the box to the bottom frame
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

    # Crop the image so that we only get the letter/word
    # Structure image[rows, col]
    crop = otsu[(hImg - h):(hImg - y), x:w]

    # Height and width of the cropped image
    hBox, wBox = crop.shape

    # Checks if the crop is too small or too large
    if hBox > (1 / hBox * 100) and wBox > (1 / hBox * 100):
        # appends the crop to the array
        segmentedLetters.append(crop)
        print("Successfully appended letter")

        # Creates a blue rectangle over the letter/image
        cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (255, 0, 0), 1)
    else:
        print("Image to small or to large.")

        # Creates a red rectangle over it
        cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

# Saves the image with all the rectangles
print("Saving the img was successful:", cv2.imwrite('segImage.png', img))

print(segmentedLetters)
