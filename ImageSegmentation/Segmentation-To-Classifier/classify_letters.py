from optparse import Values

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
import torch.nn as nn
from PIL import Image


# Object for letters that contain the image, the coordinates, and the classification.
class Letter:
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


class Segmentor:
    def __init__(self):
        return

    def Segment(self, image):
        # Necessary for running pytesseract
        # Info on how to get it running: https://github.com/tesseract-ocr/tesseract/blob/main/README.md
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Reads image of scroll
        img = image

        # Grayscales image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

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
            if hBox != 0 and wBox != 0:
                if hBox > (1 / hBox * 100) and wBox > (1 / hBox * 100):
                    # Saves the cropped letter as an object
                    croppedLetter = Letter(crop, x, y, w, h)

                    # appends the cropped letter to the array
                    segmentedLetters.append(croppedLetter)
                    # print("Successfully appended letter")

                    # Creates a blue rectangle over the letter/image
                    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (255, 0, 0), 1)
                else:
                    # print("Image to small or to large.")

                    # Creates a red rectangle over it
                    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)

        # Saves the image with all the rectangles
        return segmentedLetters


class Classifier:
    def __init__(self, model, input_size=100):
        self.input_size = input_size

        # Setup model
        self.model = Convolutional(input_size)
        self.model.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
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
            x, y = int((100 / 2)) - int(image.width / 2), int(100) - int(image.height)
            new_image.paste(image, (x, y))

            #  Converts back into numpy array
            np_image = np.array(new_image) / 255
            image_batch[i] = np_image
        return image_batch

    def Classify(self, letters):

        images = self.__LoadImages(letters)

        # Convert the numpy arrays into tensors
        images = torch.from_numpy(images).float()

        # Fix the shape of the array
        images = images.unsqueeze(1)

        # Predict
        results = self.model(images)

        # Convert the predictions to a numpy array
        results = results.detach().numpy()

        for i, result in enumerate(results):
            confidence = np.argmax(result)
            prediction = self.classes[confidence]
            letters[i].AddLabel(prediction, confidence)

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
