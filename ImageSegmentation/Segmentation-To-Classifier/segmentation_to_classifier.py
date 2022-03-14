# OpenCV
import cv2
import pytesseract
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MetaImage():
    def __init__(self):
        # setup goes here
        return
    
    def AddLabel(self, label):
        # label function
        return 

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
        return segmentedLetters
        

class Classifier():
    def __init__(self, model):

        # Setup model
        self.model = Convolutional()
        self.model.load_state_dict(torch.load(model))
        self.model.eval()

        # Setup classes
        self._classes = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED', 'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']
    
    def Classify(self, images):
        # Ensure that the array is a numpy array
        images = np.asarray(images)

        # Convert the numpy arrays into tensors
        images = torch.from_numpy(images).float()
    
        # Fix the shape of the array
        images = images.unsqueeze(1)

        # Predict
        confidenseValues = self.model(images)

        # Convert the predictions to a numpy array
        confidenseValues = confidenseValues.detach().numpy()
        # Get the highest confidense value
        predictions = list()
        for confidenseValue in confidenseValues:
            highestConfidense = np.argmax(confidenseValue)
            predictions.append(self._classes[highestConfidense])

        return predictions

class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        # Convolutional layers and Max pooling with activation functions
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        # Fully connected layer with activation functions
        self.fullyconnected = nn.Sequential(
            nn.Linear(16 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 22)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.fullyconnected(x)
        return x