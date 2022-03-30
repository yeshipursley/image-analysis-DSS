from optparse import Values
import torch
import torch.nn as nn
import torch.nn.functional as nnf

import cv2, pytesseract, math
import numpy as np
from PIL import Image

from image_straighten import deskew, unshear

import matplotlib.pyplot as plt


# Object for letters that contain the image, the coordinates, and the classification.
class Letter():
    def __init__(self, image, x, y, w, h):
        self.image = image
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = None
        self.confidence = None

    def AddLabel(self, label, confidence):
        self.label = label
        self.confidence = confidence


# Straightens the letters in an image
def image_straighten(image):
    img = image

    thresh = cv2.threshold(img, 127, 255, 1)[1]

    deskew(thresh)
    sheared_img = unshear(thresh)

    ret, thresh = cv2.threshold(sheared_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh


# Returns the confidence value of a letter as a boolean.
def classLetterChecker(image):
    _, confidence_value = Classifier.SimplyClassify(image)
    if confidence_value > 80:
        return True
    else:
        return False


# Splits a word into letters
def word_splitter(word):
    image = image_straighten(word)

    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Skeletonizes the image
    while True:
        open = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, open)

        eroded = cv2.erode(image, element)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        if cv2.countNonZero(image) == 0:
            break

    hSkel, wSkel = skel.shape

    # array for sum of vertical pixels
    amountVertPixels = []

    # counts the sum of vertical pixels in image
    for i in range(wSkel):
        colPixels = int((sum(skel[:, i])) / 255)
        amountVertPixels.append(colPixels)

    minLetterWidth = 15
    segPoints = []

    index = len(amountVertPixels) - 1

    # Finds the segmentation points with the sum of vertical pixels
    while index > 0:
        if amountVertPixels[index] > 0:
            for j in range(index, -1, -1):
                if amountVertPixels[j] < 1:
                    if amountVertPixels[j - 1] < 1:
                        if (index - j) > minLetterWidth:
                            segPoints.append(j)
                            index = j
                            break
                        else:
                            index = j
                            break
                if amountVertPixels[j] > 5:
                    if (index - j) > minLetterWidth:
                        segPoints.append(j)
                        index = j
                        break
        index -= 1

    # crops the letter based on the segmentation point
    imageID = 1
    segmentationIndex = len(amountVertPixels)
    segmentedLettersInWord = []
    for i in segPoints:
        extend_image = 2
        # if the segmentation is on the right side of the image
        if segmentationIndex == len(amountVertPixels):
            cropped_image = word[:, i:segmentationIndex]
            while classLetterChecker(cropped_image):
                # extends the image to the left until sufficient classification value
                cropped_image = word[:, i - extend_image:segmentationIndex]
                extend_image += 2
            cropped_letter = Letter(cropped_image, i, None, segmentationIndex, None, None)
            segmentedLettersInWord.append(cropped_letter)
        # if the segmentation point is on the left side of the image
        elif i < 2:
            cropped_image = word[:, 0:segmentationIndex]
            while classLetterChecker(cropped_image):
                # extends the image to the right until sufficient classification value
                cropped_image = word[:, 0:segmentationIndex + extend_image]
                extend_image += 2
            cropped_letter = Letter(cropped_image, i, None, segmentationIndex, None, None)
            segmentedLettersInWord.append(cropped_letter)
        # if the segmentation point is in the middle of the image
        else:
            croppedImage = word[:, i - 2:segmentationIndex + 2]
            while classLetterChecker(croppedImage):
                # Makes sure the crop doesn't go out of bounds
                if i - extend_image < 0:
                    start_crop_value = 0
                else:
                    start_crop_value = i - extend_image
                # extends the crop on both sides
                cropped_image = word[:, start_crop_value:segmentationIndex + extend_image]
                extend_image += 2
            cropped_letter = Letter(cropped_image, i, None, segmentationIndex, None, None)
            segmentedLettersInWord.append(cropped_letter)
        segmentationIndex = i
        imageID += 1
    return segmentedLettersInWord


class Segmentor():
    def __init__(self):
        return

    def Segment(self, image):
        # Necessary for running pytesseract
        # Info on how to get it running: https://github.com/tesseract-ocr/tesseract/blob/main/README.md
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Reads image of scroll
        img = image

        # Grayscales image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # otsu thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoises the otsu image
        deNoiseOtsu = cv2.fastNlMeansDenoising(otsu, h=60.0, templateWindowSize=7, searchWindowSize=21)

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
                # If the segment is larger than 40 pixels wide
                if wBox > 40:
                    for i in word_splitter(crop):
                        # Saves each segmented letter as a Letter object with the correct coordinate values
                        croppedLetter = Letter(i, x + i.x, y, x + i.w, h, None)

                        # appends the cropped letter to the array
                        segmentedLetters.append(croppedLetter)
                # Saves each segmented letter as a Letter object with the correct coordinate values
                croppedLetter = Letter(crop, x, y, w, h)

                # appends the cropped letter to the array
                segmentedLetters.append(croppedLetter)
            else:
                # print("Image to small or to large.")

                # Creates a red rectangle over it
                cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

        # Saves the image with all the rectangles
        return segmentedLetters


class Classifier():
    def __init__(self, model, input_size=100):
        self.input_size = input_size

        # Setup model
        self.model = Convolutional(input_size)
        self.model.load_state_dict(torch.load(model))
        self.model.eval()

        # Setup classes
        self.classes = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED',
                        'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']

    def __LoadImages(self, letters):
        image_batch = np.zeros((len(letters), self.input_size, self.input_size))
        for i, letter in enumerate(letters):
            # Load image from array
            image = Image.fromarray(letter.image)

            # Fix the dimensions of the image
            new_image = Image.new(image.mode, (100, 100), 255)
            x, y = int((100 / 2)) - int(image.width / 2), int(100/2) - int(image.height/2)
            new_image.paste(image, (x, y))

            #  Converts back into numpy array
            np_image = np.array(new_image) / 255
            image_batch[i] = np_image

        return image_batch
    
    def SimplyClassify(self, image):
        # Fix the dimensions of the image
        image = Image.fromarray(image)
        new_image = Image.new(image.mode, (100, 100), 255)
        x, y = int((100 / 2)) - int(image.width / 2), int(100/2) - int(image.height/2)
        new_image.paste(image, (x, y))
        np_image = np.array(new_image) / 255

        # Convert the numpy arrays into tensors
        image = torch.from_numpy(np_image).float()

        
        # Fix the shape of the array
        image = image.unsqueeze(0).unsqueeze(0)
        # Predict
        result = self.model(image)
        result = result[0]
        # Convert the predictions to a numpy array
        
        result = nnf.softmax(result, dim=0)
        result = result.detach().numpy()
        confidence = np.argmax(result)
        prediction = self.classes[confidence]
        return prediction, result[confidence] * 100


    def Classify(self, letters):

        images = self.__LoadImages(letters)

        # Convert the numpy arrays into tensors
        images = torch.from_numpy(images).float()

        # Fix the shape of the array
        images = images.unsqueeze(1)

        # Predict
        results = self.model(images)

        # Convert the predictions to a numpy array

        for i, result in enumerate(results):
            result = nnf.softmax(result, dim=0)
            result = result.detach().numpy()

            confidence = np.argmax(result)
            prediction = self.classes[confidence]
            letters[i].AddLabel(prediction, math.trunc(result[confidence] * 100))

        return letters


class Convolutional(nn.Module):
    def __init__(self, input_size):
        super(Convolutional, self).__init__()
        # Convolutional layers and Max pooling with activation functions
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layer with activation functions
        size = int((input_size / 4) - 3)
        self.fullyconnected = nn.Sequential(
            nn.Linear(16 * size * size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 22)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.fullyconnected(x)
        return x
