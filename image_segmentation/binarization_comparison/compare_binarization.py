# OpenCV
import cv2
# For changing the directory so that we can save images in a different folder
import os

# Reads image of scroll
img = cv2.imread('PLACEHOLDER')

# Grayscales image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# adaptive thresholding Gaussian - Binary
gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 15)

# adaptive thresholding Mean - Binary
mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 15)

# otsu thresholding - Binary
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# median blur with Otsu
median = cv2.medianBlur(otsu, 5)

# Change the current directory
# to specified directory
# for placing the results
os.chdir('PLACEHOLDER')

# Saves image
print("Saving the gauss image was successful:", cv2.imwrite('gauss.png', gauss))
print("Saving the mean image was successful:", cv2.imwrite('mean.png', mean))
print("Saving the otsu image was successful:", cv2.imwrite('otsu.png', otsu))
print("Saving the median image was successful:", cv2.imwrite('median_blur_otsu.png', median))
