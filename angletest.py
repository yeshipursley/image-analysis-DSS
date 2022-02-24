import matplotlib.pyplot as plt
import cv2
import numpy as np

image = 'LAMED_82.png'
image = cv2.imread(image)
for i in range(11):
    angle = 45 * (i/10)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    plt.subplot(1,11,i+1)
    plt.title(str(angle))
    plt.imshow(rotated_image)

plt.show()