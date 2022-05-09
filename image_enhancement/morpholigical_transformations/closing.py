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

# Getting the image path and save path from the command line
image_path = None
save_path = None

for arg in args:
    if arg == "-input":
        image_path = args[args.index("-input") + 1]
    elif arg == "-output":
        save_path = args[args.index("-output") + 1]

# Checking if something is wrong with the arguments.
if not image_path or not save_path:
    print("You must specify both image path and save path.")
    sys.exit(2)

if not isinstance(image_path, str) or not isinstance(save_path, str):
    print("The image path and save path must both be strings.")
    sys.exit(2)

if not os.path.exists(image_path):
    print("The image path you have specified does not exits.")
    sys.exit(2)

save_arr = save_path.split('\\')

save_path_base = ""
i = 0
while i < len(save_arr):
    if i == len(save_arr) - 1:
        break
    elif i == 0:
        save_path_base += save_arr[i]
    else:
        save_path_base += "\\" + save_arr[i]
    i += 1

if not os.path.exists(save_path_base):
    print("The save path you have specified does not exits.")
    sys.exit(2)

# Read the binarized image using imageio
img = img.imread(image_path)

# Inverting the image
inverted_img = cv.bitwise_not(img)

# Creates an elliptical kernel
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

# Performs closing on the image
closed_img = cv.morphologyEx(inverted_img, cv.MORPH_CLOSE, kernel)

# Inverts the image back
inverted_back = cv.bitwise_not(closed_img)

# Saving the image that has gone through closing
save_img = Image.fromarray(inverted_back)
save_img.save(save_path)

print(
    "\nImage " + image_path + " has successfully gone through closing and has been stored here: " + save_path)