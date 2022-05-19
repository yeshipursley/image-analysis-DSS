import math

import cv2
import numpy as np
import pytesseract
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
import image_straighten as img_straighten


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


# Returns the confidence value of a letter as a boolean.
def class_letter_checker(image):
    classifier = Classifier("default.model")
    _, confidence_value = classifier.SimplyClassify(image)
    return confidence_value


# Straightens the letters in an image
# Source: https://github.com/RiteshKH/Cursive_handwriting_recognition/blob/master/image-straighten.py
# Date: 11.05.2022
def image_straighten(image):
    img = image

    thresh = cv2.threshold(img, 127, 255, 1)[1]

    img_straighten.deskew(thresh)
    sheared_img = img_straighten.unshear(thresh)

    ret, thresh = cv2.threshold(sheared_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh


# Removes the white space over a letter
def image_cropper(img):
    h_img, w_img = img.shape
    if h_img >= 1 and w_img >= 1:
        edged = cv2.Canny(img, 30, 200)

        coords = cv2.findNonZero(edged)
        x, y, w, h = cv2.boundingRect(coords)
        crop = img[y:y + h, :]

        return crop
    else:
        return None


# Skeletonizes the image
# Source: https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
# Date: 11.05.2022
def skeletonize(image):
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

    return skel


# calculates the segmentation points based on the array with the sum of the vertical pixels in an image
def segmentation_point_finder(amount_vert_pixels, min_letter_width):
    seg_points = []

    index = len(amount_vert_pixels) - 1

    # Finds the segmentation points with the sum of vertical pixels
    # iterates through the list backwards
    while index > 0:
        if amount_vert_pixels[index] > 0:
            for j in range(index, -1, -1):
                # if no vertical pixels are found
                if amount_vert_pixels[j] < 1:
                    # if no vertical pixels are found in the next element
                    if amount_vert_pixels[j - 1] < 1 or j - 1 < 0:
                        # this prevents segmentations from being two close
                        if (index - j) > min_letter_width:
                            seg_points.append(j)
                            index = j
                            break
                        else:
                            index = j
                            break
                # if detected a vertical line
                if amount_vert_pixels[j] > 5:
                    if (index - j) > min_letter_width:
                        seg_points.append(j)
                        index = j
                        break
        index -= 1
    return seg_points


# Splits an image with multiple letters into multiple images each containing one letter.
# The function uses the segmentation points as a baseline for the segments.
def word_cropper(seg_points, amount_vert_pixels, word, min_letter_width):
    segmented_letters_in_word = []
    segmentation_index = len(amount_vert_pixels) - 1

    for i in seg_points:

        extend_image = 2
        out_of_bounds = False

        # if the segmentation is on the far right side of the image
        if segmentation_index > len(amount_vert_pixels) - 5:
            cropped_image = word[:, i:len(amount_vert_pixels)]
            cropped_image = image_cropper(cropped_image)

            confidence_value = 0
            best_extend_image = 0
            while True:
                new_confidence_value = class_letter_checker(cropped_image)
                if new_confidence_value > 60:
                    break
                # checks if we have extended the cropped image too far or if we have gone out of bounds
                # too far is defined here as more than half the min_letter_width
                elif extend_image > min_letter_width / 2 or out_of_bounds is True:
                    cropped_image = word[:, i - best_extend_image:len(amount_vert_pixels)]
                    cropped_image = image_cropper(cropped_image)
                    break
                else:
                    # checks if the new confidence value is higher than the current hightest one
                    if confidence_value < new_confidence_value:
                        confidence_value = new_confidence_value
                        best_extend_image = extend_image
                    if i - extend_image < 0:
                        out_of_bounds = True
                    else:
                        # extends the image to the left until sufficient classification value
                        cropped_image = word[:, i - extend_image:len(amount_vert_pixels)]
                        cropped_image = image_cropper(cropped_image)
                        extend_image += 2
            final_extend_image_left = i - best_extend_image
            if final_extend_image_left < 0:
                final_extend_image_left = 0

            if cropped_image is not None:            
                # Checks if the letter is thinner than the min_letter_width
                if len(amount_vert_pixels) - final_extend_image_left >= min_letter_width:
                    cropped_letter = Letter(cropped_image, final_extend_image_left, None, segmentation_index, None)
                    segmented_letters_in_word.append(cropped_letter)

        # if the segmentation point is on the left side of the image
        elif i < 4:
            cropped_image = word[:, 0:segmentation_index]
            cropped_image = image_cropper(cropped_image)

            confidence_value = 0
            best_extend_image = 0
            while True:
                new_confidence_value = class_letter_checker(cropped_image)
                if new_confidence_value > 60:
                    # finished
                    break
                # checks if we have extended the cropped image too far or if we have gone out of bounds
                # too far is defined here as more than half the min_letter_width
                elif extend_image > min_letter_width / 2 or out_of_bounds is True:
                    cropped_image = word[:, 0:segmentation_index + best_extend_image]
                    cropped_image = image_cropper(cropped_image)
                    break
                else:
                    # Saves the best confidence value and its extend_image value
                    if confidence_value < new_confidence_value:
                        confidence_value = new_confidence_value
                        best_extend_image = extend_image
                    if segmentation_index + extend_image > len(amount_vert_pixels):
                        out_of_bounds = True
                    else:
                        # extends the image to the right until sufficient classification value
                        cropped_image = word[:, 0:segmentation_index + extend_image]
                        cropped_image = image_cropper(cropped_image)
                        extend_image += 2

            final_extend_image_right = segmentation_index + best_extend_image
            if cropped_image is not None:
                _, width_cropped_image = cropped_image.shape
    
                # Checks if the letter is thinner than the min_letter_width
                if width_cropped_image >= min_letter_width:
                    cropped_letter = Letter(cropped_image, i, None, final_extend_image_right, None)
                    segmented_letters_in_word.append(cropped_letter)

        # if the segmentation point is in the middle of the image
        else:
            cropped_image = word[:, i - extend_image:segmentation_index + extend_image]
            cropped_image = image_cropper(cropped_image)

            confidence_value = 0
            best_extend_image = 0
            while True:
                new_confidence_value = class_letter_checker(cropped_image)
                if new_confidence_value > 60:
                    # finished
                    break
                # checks if we have extended the cropped image too far or if we have gone out of bounds
                # too far is defined here as more than half the min_letter_width
                elif extend_image > min_letter_width or out_of_bounds is True:
                    cropped_image = word[:, i - best_extend_image:segmentation_index + best_extend_image]
                    cropped_image = image_cropper(cropped_image)
                    break
                else:
                    # Saves the best confidence value and its extend_image value
                    if confidence_value < new_confidence_value:
                        confidence_value = new_confidence_value
                        best_extend_image = extend_image
                    if i - extend_image < 0 or segmentation_index + extend_image > len(amount_vert_pixels):
                        out_of_bounds = True
                    else:
                        # extends the crop on both sides
                        cropped_image = word[:, i - extend_image:segmentation_index + extend_image]
                        cropped_image = image_cropper(cropped_image)
                        extend_image += 2

            final_extend_image_left = i - best_extend_image
            final_extend_image_right = segmentation_index + best_extend_image

            if cropped_image is not None:
                _, width_cropped_image = cropped_image.shape

                if width_cropped_image >= min_letter_width:
                    cropped_letter = Letter(cropped_image, final_extend_image_left, None, final_extend_image_right, None)
                    segmented_letters_in_word.append(cropped_letter)

        # Changes the index to the segmentation point
        segmentation_index = i

    return segmented_letters_in_word


# Splits a word into letters
def word_splitter(word):
    # straightens the letter/letters in the image
    image = image_straighten(word)

    # skeletonizes the image
    skel = skeletonize(image)

    # gets the height and width of the skeletonized image
    h_skel, w_skel = skel.shape

    # array for sum of vertical pixels
    amount_vert_pixels = []

    # counts the sum of vertical pixels in image
    for i in range(w_skel):
        col_pixels = int((sum(skel[:, i])) / 255)
        amount_vert_pixels.append(col_pixels)

    # We have set this value as 12 by looking at the letter "zayin", which is one of the thinnest letters, and checked
    # how many pixels wide it is in the image. The size of the letters can be different in other images, so it is
    # important to check if this constant applies to your image.
    min_letter_width = 12

    # calculates the segmentation points in the image from the amount_vert_pixels array
    seg_points = segmentation_point_finder(amount_vert_pixels, min_letter_width)

    # if there are no segmentation points we add one at the far
    # left side of the image so that the whole image gets segmented.
    if not seg_points:
        seg_points.append(0)

    # if the last segmentation point is smaller than half the minimum letter width and there more than 1 segmentation
    # points, it changes the last segmentation point to 0 since it is most likely not segmenting the entire letter
    elif seg_points[-1] <= min_letter_width / 2 and len(seg_points) > 1:
        seg_points[-1] = 0

    # if the last segmentation point is larger than half of the minimum letter width, it appends a 0 to the
    # segmentation points array. This is done to include the last letter.
    elif seg_points[-1] > min_letter_width / 2:
        seg_points.append(0)

    # Crops the image of the word using the seg_points array, amout_vert_pixels array and the min_letter_width
    segmented_letters_in_word = word_cropper(seg_points, amount_vert_pixels, word, min_letter_width)

    # Reverses the array so that the letters are in the right order
    # Most of the word splitter is performed reading the letters from right to left, which is why we need to reverse it
    segmented_letters_correct = segmented_letters_in_word[::-1]

    return segmented_letters_correct


class Segmentor:
    def __init__(self):
        return

    def segment_letters(self, image):
        # Necessary for running pytesseract
        # Info on how to get it running: https://github.com/tesseract-ocr/tesseract/blob/main/README.md
        pytesseract.pytesseract.tesseract_cmd = r'tesseract\tesseract.exe'

        # Crops the images around the letters/words
        # Saves the height and width of the images
        h_img, w_img = image.shape

        # array for the segmented letters
        segmented_letters = []

        # Makes a box around each letter/word on the scroll
        boxes = pytesseract.image_to_boxes(image, lang="heb")

        # For each box
        for b in boxes.splitlines():
            # Splits the values of the box into an array
            b = b.split(' ')
            # Save the coordinates of the box:
            # x = Distance between the left side of the box to the left frame of the image
            # y = Distance between the top of the box to the bottom frame of the image
            # w = Distance between the right side of the box to the left frame of the image
            # h = Distance between the bottom of the box to the bottom frame of the image
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

            # Crop the image so that we only get the letter/word
            # Structure image[rows, col]
            crop = image[(h_img - h):(h_img - y), x:w]

            # Height and width of the cropped image
            h_box, w_box = crop.shape

            # Checks if the crop is too small or too large
            if h_box != 0 and w_box != 0:
                # Checks if the crop is too small or too large
                if h_box > (1 / h_box * 100) and w_box > (1 / h_box * 100):
                    # If the segment is larger than 30 pixels wide
                    if w_box > 30:

                        # checks if the box is a large letter
                        if class_letter_checker(crop) > 90:
                            # Saves each segmented letter as a Letter object with the correct coordinate values
                            cropped_letter = Letter(crop, x, y, w, h)

                            # appends the cropped letter to the array
                            segmented_letters.append(cropped_letter)
                        else:
                            for i in word_splitter(crop):
                                # Saves each segmented letter as a Letter object with the correct coordinate values
                                cropped_letter = Letter(i.image, x + i.x, y, x + i.w, h)

                                # appends the cropped letter to the array
                                segmented_letters.append(cropped_letter)

                    # Found single letter
                    else:
                        # Saves each segmented letter as a Letter object with the correct coordinate values
                        cropped_letter = Letter(crop, x, y, w, h)

                        # appends the cropped letter to the array
                        segmented_letters.append(cropped_letter)
        # Saves the image with all the rectangles
        return segmented_letters

    # Method that is run if the background in the image isn't varied
    def segment_clear_background(self, image):
        # Reads image of scroll
        img = image

        # Grayscales image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Does adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(80, 80))
        equalized = clahe.apply(gray)

        # otsu thresholding
        _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # closing
        inverted_img = cv2.bitwise_not(otsu)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closedImg = cv2.morphologyEx(inverted_img, cv2.MORPH_CLOSE, kernel)

        inverted_back = cv2.bitwise_not(closedImg)

        # Denoises the closed otsu image
        denoise_otsu = cv2.fastNlMeansDenoising(inverted_back, h=60.0, templateWindowSize=7, searchWindowSize=21)

        return self.segment_letters(denoise_otsu)

    # Method that is run if the background in the image is varied
    def segment_varied_background(self, image):
        # Reads image of scroll
        img = image

        # Grayscales image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Adaptive binarization
        binarize_im = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                            thresholdType=cv2.THRESH_BINARY, blockSize=39, C=15)

        # Opening
        inverted_img = cv2.bitwise_not(binarize_im)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        opened_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)

        inverted_back = cv2.bitwise_not(opened_img)

        # Closing
        inverted_img = cv2.bitwise_not(inverted_back)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closedImg = cv2.morphologyEx(inverted_img, cv2.MORPH_CLOSE, kernel)

        inverted_back = cv2.bitwise_not(closedImg)

        # Noise removal
        denoise_otsu = cv2.fastNlMeansDenoising(inverted_back, h=60.0, templateWindowSize=7, searchWindowSize=21)

        return self.segment_letters(denoise_otsu)


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
            if letter.image is not None:
                # Load image from array
                image = Image.fromarray(letter.image)

                # Fix the dimensions of the image
                new_image = Image.new(image.mode, (100, 100), 255)
                # For sigmoid+.model
                # x, y = int((100 / 2)) - int(image.width / 2), int(100) - int(image.height)
                # For default.model
                x, y = int((100 / 2)) - int(image.width / 2), int(100 / 2) - int(image.height / 2)
                new_image.paste(image, (x, y))

                #  Converts back into numpy array
                np_image = np.array(new_image) / 255
                image_batch[i] = np_image

        return image_batch

    def SimplyClassify(self, image):
        # Fix the dimensions of the image
        image = Image.fromarray(image)
        new_image = Image.new(image.mode, (100, 100), 255)
        x, y = int((100 / 2)) - int(image.width / 2), int(100 / 2) - int(image.height / 2)
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
    def __init__(self, size):
        super(Convolutional, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        size = int((size / 4) - 3)
        self.fullyconnected = nn.Sequential(
            nn.Linear(16 * size * size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid(),

        )

        self.fc3 = nn.Linear(128, 22)

    def forward(self, x):
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.fullyconnected(x)
        x = self.fc3(x)
        return x


class Tester():
    def __init__(self):
        return

    # Takes the letters in the YOLO format from the txt file and crops the letters based on the letters coordinates.
    # Appends the cropped letters to an array
    # Source: https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
    # Date: 11.05.2022
    def yolo_to_crop(self, original_image):
        # Grayscales image
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # otsu thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        dh, dw = otsu.shape

        fl = open("YOLO FILE PLACEHOLDER", 'r')
        data = fl.readlines()
        fl.close()

        manually_segmented_letters = []

        for dt in data:
            # Split string to float
            _, x, y, w, h = map(float, dt.split(' '))

            # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
            # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
            # Date: 11.05.2022
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            # crops the image based on the coordinates
            crop = otsu[t:b, l:r]

            if crop.size:
                # Creates a letter object for the image
                ground_truth_letter = Letter(crop, l, t, r, b)

                # appends the letter to the array
                manually_segmented_letters.append(ground_truth_letter)

        return manually_segmented_letters

    # checks if two images overlap each other
    # Source: https://www.baeldung.com/java-check-if-two-rectangles-overlap
    # Date: 11.05.2022
    def is_overlapping(self, first_image, second_image):
        if first_image.y > second_image.h or first_image.h < second_image.y:
            return False
        if first_image.w < second_image.x or first_image.x > second_image.w:
            return False
        # We were getting this error in the IOU method:
        # error: (-215:Assertion failed) !_img.empty() in function 'cv::imwrite'
        # To prevent this we check if the image has a size.
        if first_image.image.size and second_image.image.size:
            return True
        else:
            return False

    # calculates the iou of the ground truth letters and the automatically segmented letters
    def IOU(self, ground_truth, segmented_letters, image):

        h_img, _, _ = image.shape
        # makes the coordinates compatible with test
        for i in segmented_letters:
            i.h = h_img - i.h
            i.y = h_img - i.y

        iou_scores = []

        # Iterates through the arrays and calculates the iou of the two letters
        for ground in ground_truth:
            for auto in segmented_letters:

                # checks if the letters are overlapping
                if self.is_overlapping(ground, auto):
                    # Source: https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
                    # Date: 11.05.2022
                    x_inter1 = max(ground.x, auto.x)
                    y_inter1 = max(ground.y, auto.y)
                    x_inter2 = min(ground.w, auto.w)
                    y_inter2 = min(ground.h, auto.h)

                    width_inter = abs(x_inter2 - x_inter1)
                    height_inter = abs(y_inter2 - y_inter1)

                    area_inter = width_inter * height_inter
                    width_box1 = abs(ground.w - ground.x)
                    height_box1 = abs(ground.h - ground.y)
                    width_box2 = abs(auto.w - auto.x)
                    height_box2 = abs(auto.h - auto.y)

                    area_box1 = width_box1 * height_box1
                    area_box2 = width_box2 * height_box2

                    area_union = area_box1 + area_box2 - area_inter

                    iou = area_inter / area_union

                    iou_scores.append(iou)

                    print("IOU: " + str(iou))

        # Calculates the average iou_score
        print("Average IOU score: " + str(np.average(iou_scores)))
