import cv2, os, random, re
from matplotlib.style import available
import numpy as np
import matplotlib.pyplot as plt
from torch import rand

def main():
    max_files = 350
    folder_path = "raw_data/tabita_dataset"

    transformations = [rotate, morphological, shear]
    for subdir, dirs, files in os.walk(folder_path):
        if len(files) <= 0:
            continue

        label = re.split(r'\\', subdir)[1]

        total_num_files = len(files)
        while total_num_files < max_files:
            # pick random image
            r_i = random.randrange(0, len(files))
            image = cv2.imread(subdir+'/'+files[r_i])
            og_image = image
            image = (255-image)
        
            # to random transformations on that image
            transform_num = random.randrange(1, len(transformations))
            for i in range(transform_num):
                image = transformations[random.randrange(0, len(transformations))](image)
            
            image = (255-image)

            # save image
            path = subdir + '/' + label + str(total_num_files) + '.png'
            print(f'Saving to: {path}')
            cv2.imwrite(path, image)
            total_num_files += 1

def rotate(image):
    angle = 15
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)

def morphological(image):
    kernel = np.ones((3,3), np.uint8)
    if bool(random.getrandbits(1)):
        return cv2.erode(image, kernel)
    else:
        return cv2.dilate(image, kernel)

def skew(image):
    # not sure what operations to do here
    print("Skewing image")

def shear(image):
    W, H, _ = image.shape
    M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
    M2[0,2] = -M2[0,1] * W/2
    M2[1,2] = -M2[1,0] * H/2
    return cv2.warpAffine(image, M2, image.shape[1::-1])


if __name__ == "__main__":
    main()