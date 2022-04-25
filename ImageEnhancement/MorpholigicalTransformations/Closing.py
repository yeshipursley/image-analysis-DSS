import imageio as img
from PIL import Image
import cv2 as cv
import sys, getopt, os

# Getting the arguments from the command line.
argv = sys.argv
try:
    _, args = getopt.getopt(argv, "hm:e:", ["input", "output"])
except:
    print("Error")
    sys.exit(2)

imagePath = None
savePath = None

for arg in args:
    if arg == "-input":
        imagePath = args[args.index("-input") + 1]
    elif arg == "-output":
        savePath = args[args.index("-output") + 1]

# Checking if something is wrong with the arguments.
if not imagePath or not savePath:
    print("You must specify both image path and save path.")
    sys.exit(2)

if not isinstance(imagePath, str) or not isinstance(savePath, str):
    print("The image path and save path must both be strings.")
    sys.exit(2)

if not os.path.exists(imagePath):
    print("The image path you have specified does not exits.")
    sys.exit(2)

saveArr = savePath.split('\\')

savePathBase = ""
i = 0
while i < len(saveArr):
    if i == len(saveArr) - 1:
        break
    elif i == 0:
        savePathBase += saveArr[i]
    else:
        savePathBase += "\\" + saveArr[i]
    i += 1

if not os.path.exists(savePathBase):
    print("The save path you have specified does not exits.")
    sys.exit(2)

# Read the grayscale image using imageio
img = img.imread(imagePath)

# Inverting the image
invertedImg = cv.bitwise_not(img)

# Creates an elliptical kernel
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

# Performs closing on the image
closedImg = cv.morphologyEx(invertedImg, cv.MORPH_CLOSE, kernel)

# Inverts the image back
invertedBack = cv.bitwise_not(closedImg)

# Saving the image that has gone through closing
saveImg = Image.fromarray(invertedBack)
saveImg.save(savePath)

print(
    "\nImage " + imagePath + " has successfully gone through closing and has been stored here: " + savePath)