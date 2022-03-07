import cv2
import numpy

image = cv2.imread('Data/tabita_dataset/alef/alef_col01_90_binarized.png')
print(image)
print("Numpy")
test = numpy.array(image)
print(test)