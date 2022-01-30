import os
from PIL import Image
import re
import csv

import matplotlib

import numpy as np
import matplotlib.pyplot as plt
# Create a CSV file with all the resized images
path = 'datasets/test/'
# num_files = len(os.listdir(path))
# labels = ['alef', 'het', 'mem', 'shin']
# whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# dataset = list()
# for i, file in enumerate(os.listdir(path)):
#     label = ''.join(filter(whitelist.__contains__, file))
#     image = Image.open(path+file).convert('L') # Open the image
#     index = 0
#     if(label == 'alef'):
#         index = 0
#     elif(label == 'het'):
#         index = 1
#     elif(label == 'mem'):
#         index = 2
#     elif(label == 'shin'):
#         index = 3
#     dataset.append([file, index])
    
# # Write to csv
# f = open('datasets/train.csv', 'w')
# writer = csv.writer(f)
# writer.writerows(dataset)
# f.close()
    
# for i, file in enumerate(os.listdir(path)):
#     print(file)
#     image = Image.open(path+file).convert('L')
#     image = image.resize((32,32))
#     image.save(path+file)

# for i, file in enumerate(os.listdir(path)):
#     og_image = Image.open(path+file).convert('L')
#     new_image = Image.new(og_image.mode, (64,64), 255)
#     new_image.paste(og_image, (00,0))
#     new_image.save(path+file[:-4]+'_1.png')
#     new_image = Image.new(og_image.mode, (64,64), 255)
#     new_image.paste(og_image, (32,0))
#     new_image.save(path+file[:-4]+'_2.png')
#     new_image = Image.new(og_image.mode, (64,64), 255)
#     new_image.paste(og_image, (32,32))
#     new_image.save(path+file[:-4]+'_3.png')
#     new_image = Image.new(og_image.mode, (64,64), 255)
#     new_image.paste(og_image, (0,32))
#     new_image.save(path+file[:-4]+'_4.png')
