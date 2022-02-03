import enum
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv

# Dont change
new_path = "datasets"
folder_path = "characters"

# check for files
print("Checking if files and directories exist")
if not os.path.isfile('datasets/test.csv'):
    f = open('datasets/test.csv', 'x')
    f.close()

if not os.path.isfile('datasets/train.csv'):
    f = open('datasets/train.csv', 'x')
    f.close()

if not os.path.isdir('datasets/train'):
    os.mkdir('datasets/train')

if not os.path.isdir('datasets/test'):
    os.mkdir('datasets/test')

# for each subdirectory in folder
test_rows = list()
train_rows = list()

current_letter = 0
classes = ['alef', 'ayin', 'bet', 'dalet', 'gimel', 'het', 'he', 'kaf', 'lamed', 'mem', 'nun', 'pe', 'qof', 'resh', 'samekh', 'shin', 'tav', 'tet', 'tsadi', 'vav', 'yod', 'zayin']

print("Looping through characters:")
for subdir, dirs, files in os.walk(folder_path):
    num_files = len(files)
    print("Current subdirectory: \n"+ subdir)
    if(num_files == 0):
        continue

    split = int(num_files * 0.8)

    label = subdir[len(folder_path)+1:-10]
    for i, file in enumerate(files):
        image_path = subdir + "\\" + file     
        ## Open image              
        image = Image.open(image_path).convert('L')  
        ## Create empty bigger image           
        #new_image = Image.new(image.mode, (64,64), 255)
        ## Calculate center position         
        #x, y = int(image.width/2), int(image.height/2)          
        ## Paste image in the middle of the emtpy image
        #new_image.paste(image, (x,y))

        new_image = image.resize((32,32))

        # Save image into either training or testing folders
        image_name = label + str(i) + ".png"
        if(i < split):
            new_image.save("datasets/train/" + image_name)
            train_rows.append((image_name, classes.index(label)))
        else:
            new_image.save("datasets/test/" + image_name)
            test_rows.append((image_name, classes.index(label)))

# Write to csv files
with open('datasets/train.csv', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(train_rows)

with open('datasets/test.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(test_rows)