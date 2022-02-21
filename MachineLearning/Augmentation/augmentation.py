import cv2, os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import img_to_graph

def crop(image, w, h):
    center_x, center_y = image.shape[0] / 2, image.shape[1] / 2
    x = center_x - w/2
    y = center_y - h/2

    return image[int(y):int(y+h), int(x):int(x+w)]

folder_path = "MachineLearning/NeuralNetwork/datasets/images"

for subdir, dirs, files in os.walk(folder_path):
    if len(files) <= 0:
        continue

    for i, file in enumerate(files):
        image = cv2.imread(f'{subdir}\{file}')
        image = (255-image) # Invert
        image_size = image.shape # Get size
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT) # Add padding
 
        # Rotation transform
        num_rows, num_cols = image.shape[:2]
        for x in [-1, 0, 1]:
            # Rotation
            rotation_angle = 15 * x
            img_rotation = cv2.warpAffine(image, cv2.getRotationMatrix2D((num_cols/2, num_rows/2), rotation_angle, 0.6), (num_cols, num_rows))

            # Morpho stufg
            kernel = np.ones((2,2),np.uint8)
            erosion = cv2.erode(img_rotation, kernel, iterations = 1)
            dilation = cv2.dilate(img_rotation, kernel, iterations= 1)

            # Crop images
            img_rotation = crop(img_rotation, image_size[0], image_size[1])
            erosion = crop(erosion, image_size[0], image_size[1])
            dilation = crop(dilation, image_size[0], image_size[1])

            # Invert back
            img_rotation = (255 - img_rotation)
            erosion = (255 - erosion)
            dilation = (255 - dilation)

            plt.subplot(1,3,1), plt.title("Image"), plt.imshow(img_rotation)
            plt.subplot(1,3,2), plt.title("Erosion"), plt.imshow(erosion)
            plt.subplot(1,3,3), plt.title("Dilation"), plt.imshow(dilation)
            plt.show()

            #cv2.imwrite('MachineLearning/Augmentation/augmented_output/' + file[:-4] + '_r.png', img_rotation)
            #cv2.imwrite('MachineLearning/Augmentation/augmented_output/' + file[:-4] + '_e.png', erosion)
            #cv2.imwrite('MachineLearning/Augmentation/augmented_output/' + file[:-4] + '_d.png', dilation)

    
    
