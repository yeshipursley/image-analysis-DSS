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

# Read the image using imageio
img = img.imread(image_path)

# Converts the image to gray-scale if it is not already
if len(img.shape) == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
else: 
    img = img

# Perform histogram equalization on the image
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(img)

# Saving the biniarized and blured image
save_img = Image.fromarray(equalized)
save_img.save(save_path)

print(
    "\nImage " + image_path + " has successfully gone through adaptive histogram equalization and has been stored here: " + save_path)