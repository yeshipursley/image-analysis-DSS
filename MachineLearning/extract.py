import os
from re import sub
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv, random
import sys, getopt

from sklearn.svm import LinearSVR
from sklearn.utils import resample

dirname = os.path.dirname(__file__)
CLASSES = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED', 'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']

# def Method1(image):
#     ## Create empty bigger image   
#     new_size = image.width if image.width > image.height else image.height    
#     new_image = Image.new(image.mode, (new_size,new_size), 255)
    
#     ## Resize image if its bigger than the new image
#     if image.height > new_size:
#         r = image.height / image.width
#         image = image.resize((int(new_size/r),new_size), resample=Image.NEAREST)
#     elif image.width > new_size:
#         r = image.width / image.height
#         image = image.resize((new_size,(int(new_size/r))), resample=Image.NEAREST)

#     # Paste image in the middle of the emtpy image
#     x, y = int((new_size/2)) - int(image.width/2), int((new_size/2)) - int(image.height/2)  
#     new_image.paste(image, (x,y))
#     return new_image.resize((100,100), resample=Image.NEAREST)

# def Method2(image):
#     new_image = Image.new(image.mode, (100, 100), 255)
#     # x, y = int((100/2)) - int(image.width/2), int((100/2)) - int(image.height/2) 
#     x, y = int((100/2)) - int(image.width/2), int(100) - int(image.height) 
#     new_image.paste(image, (x,y))
#     return new_image

def ConvertImage(image):
    # Create a new image of larger size, size only works for the dataset where characters are not larger than 100
    new_image = Image.new(image.mode, (100, 100), 255)

    # Calculate the position of the character
    x, y = int((100/2)) - int(image.width/2), int((100/2)) - int(image.height/2) 
    
    # Paste the character on the new image
    new_image.paste(image, (x,y))
    return new_image

def Extract(folder_path, dataset_name):
    rows = [list() for x in range(22)]
    
    # for each subdirectory in folder
    for subdir, dirs, files in os.walk(folder_path):
        num_files = len(files)
        
        if(num_files == 0):
            continue

        print("Current subdirectory: "+ subdir)
        for i, file in enumerate(files):
            # Find label 
            label = subdir.upper()[5:]

            image = Image.open(subdir + "\\" + file ).convert('L')  
            new_image = ConvertImage(image)
            new_name = f'{label}_{i}.png'

            # Save stuff
            new_image.save(dirname + '\\NeuralNetwork' + "\\datasets\\"+dataset_name+"\\images\\" + new_name)
            if label.upper() == "ALEF":
                index = 0
            else:
                index = 1
            rows[index].append((new_name,index))
        
    return rows

def WriteCSV(name, rows, limit, whitelist):
    # Check for files, and create if they do not exist
    if not os.path.isfile(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\test.csv'):
        f = open(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\test.csv', 'x')
        f.close()

    if not os.path.isfile(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\train.csv'):
        f = open(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\train.csv', 'x')
        f.close()

    if not os.path.isfile(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\overview.csv'):
        f = open(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\overview.csv', 'x')
        f.close()

    for row in rows:
        if not row:
            continue
        
        label = row[0][0].split('_')[0]
        if label not in whitelist and len(whitelist) > 0:
            continue

        random.shuffle(row)
        row = row[:limit if limit != 0 else len(row)]
        l = int(len(row) * 0.8)
        train, test = row[:l], row[l:]

        with open(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\overview.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([label, str(len(row))])

        with open(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\test.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(test)
        
        with open(dirname + '\\NeuralNetwork' + '\\datasets\\'+name+'\\train.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(train)

def main(argv):
    # default values
    folder_path = ''
    whitelist = list()
    limit = 0
    name = 'default'

    try:
        opts, args = getopt.getopt(argv,"hd:l:n:", ["whitelist=", "limit=", "name="])
    except:
        # ERROR
        print("Error")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # Help 
            print("Usage: ")
            sys.exit()
        elif opt in ("-d"):
            folder_path = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("--whitelist"):
            whitelist = arg.split(',')
        elif opt in ("--limit", '-l'):
            limit = int(arg)

    if folder_path == '':
        print('Need to specify a directory')
        exit(2)

    # Check if directory exists
    if not os.path.isdir(dirname + '\\NeuralNetwork' + '\\datasets'):
            os.mkdir(dirname + '\\NeuralNetwork' + '\\datasets')

    if not os.path.isdir(dirname + '\\NeuralNetwork' + '\\datasets\\' + name):
            os.mkdir(dirname + '\\NeuralNetwork' + '\\datasets\\' + name)

    # Check if directory exists
    if not os.path.isdir(dirname + '\\NeuralNetwork' + '\\datasets\\' + name +'\\images'):
            os.mkdir(dirname + '\\NeuralNetwork' + '\\datasets\\' + name +'\\images')

    # Extract and Write to csv files
    rows = Extract(folder_path, name)
    WriteCSV(name, rows, limit, whitelist)

    print("Finished.")

if __name__ == "__main__":
   main(sys.argv[1:])

