import os
from PIL import Image
import re
import csv

# Create a CSV file with all the resized images
path = 'datasets/train/'
num_files = len(os.listdir(path))
labels = ['alef', 'het', 'mem', 'shin']

dataset = list()
for i, file in enumerate(os.listdir(path)):
    label = re.sub(r'\d+', '', file)[:-4] # Strip the name from the filepath
    image = Image.open(path+file).convert('L') # Open the image
    index = 0
    if(label == 'alef'):
        index = 0
    elif(label == 'het'):
        index = 1
    elif(label == 'mem'):
        index = 2
    elif(label == 'shin'):
        index = 3
    dataset.append([file, index])
   
# Write to csv
f = open('datasets/train.csv', 'w')
writer = csv.writer(f)
writer.writerows(dataset)
f.close()
    
