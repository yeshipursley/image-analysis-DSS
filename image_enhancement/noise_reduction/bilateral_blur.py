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

# Using bilateral blur, which is highly effective at noise removal while preserving edges.
bilateral_blur = cv.bilateralFilter(img, 9, 150, 150)

# Saving the biniarized and blured image
imgSave = Image.fromarray(bilateral_blur)
imgSave.save(save_path)

print(
    "\nImage " + image_path + " has successfully gone through bilateral blur denoisng and has been stored here: " + save_path)