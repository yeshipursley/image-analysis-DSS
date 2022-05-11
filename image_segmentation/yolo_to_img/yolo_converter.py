# Taken from https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
# Date: 25.02.2022
# Have done a few changes
import cv2
import matplotlib.pyplot as plt

# For changing the directory so that we can save images in a different folder
import os

# Reading the DSS-image
img = cv2.imread('<DSS-image>')

# Converting it to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Getting the shape of the gray-scale image
dh, dw = gray.shape

# Opening and reading the lines in the yolo-txt file and storing them in data
fl = open('<yolo-txt-file>', 'r')
data = fl.readlines()
fl.close()

# Changing the folder to the folder you want to store the crops in. May need full folder-path
os.chdir('<Folder you want to store the crops in>')

# Counter that is to be in the file name of the crops
counter = 1 

# Going through all the crops and saving them as individual images. 
# It also creates an image with rectangles over the crops that will be done. 
for dt in data:

    # Split string to float
    _, x, y, w, h = map(float, dt.split(' '))

    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    
    
    # Cropping every letter I have marked in labelImg
    crop = gray[int(t):int(b), int(l):int(r)]
    
    # The letter to use in the file-name
    letter = '<letter-name>'
    
    # the number of the column if any
    column = '<column-number>'
    
    # What the crops will be saved as in the folder you chose.
    # Should name the crops the letter that it is a crop of and some kind of counter.
    cv2.imwrite(letter + '0' + column + str(counter) + '.png', crop)
    
    # Increment the counter
    counter += 1
