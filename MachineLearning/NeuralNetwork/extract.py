import os
from re import sub
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv, random
import sys, getopt

CLASSES = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED', 'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']
def ConvertImage(image):
    ## Create empty bigger image   
    new_size = image.width if image.width > image.height else image.height    
    new_image = Image.new(image.mode, (new_size,new_size), 255)
    
    ## Resize image if its bigger than the new image
    if image.height > new_size:
        r = image.height / image.width
        image = image.resize((int(new_size/r),new_size), resample=Image.NEAREST)
    elif image.width > new_size:
        r = image.width / image.height
        image = image.resize((new_size,(int(new_size/r))), resample=Image.NEAREST)

    # Paste image in the middle of the emtpy image
    x, y = int((new_size/2)) - int(image.width/2), int((new_size/2)) - int(image.height/2)  
    new_image.paste(image, (x,y))

    return new_image.resize((32,32), resample=Image.NEAREST)

def Extract(folder_path, whitelist):
    # Check for directories
    #if not os.path.isdir('MachineLearning/NeuralNetwork/datasets/train'):
    #    os.mkdir('MachineLearning/NeuralNetwork/datasets/train')

    #if not os.path.isdir('MachineLearning/NeuralNetwork/datasets/test'):
    #    os.mkdir('MachineLearning/NeuralNetwork/datasets/test')

    rows = [list() for x in range(22)]
    
    # for each subdirectory in folder
    for subdir, dirs, files in os.walk(folder_path):
        num_files = len(files)
        
        if(num_files == 0):
            continue

        print("Current subdirectory: "+ subdir)
        for i, file in enumerate(files):
            # Find label 
            for className in CLASSES:
                if className in file.upper():
                    label = className
                    break

            if label not in whitelist and len(whitelist) > 0:
                continue

            image = Image.open(subdir + "\\" + file ).convert('L')  
            
            new_image = ConvertImage(image)
            new_name = label + str(i) + ".png"

            new_image.save("MachineLearning/NeuralNetwork/datasets/images/" + new_name)
            index = CLASSES.index(label.upper())
            rows[index].append((new_name,index))
        
    return rows

def WriteCSV(rows):
    if not os.path.isfile('MachineLearning/NeuralNetwork/datasets/test.csv'):
        f = open('MachineLearning/NeuralNetwork/datasets/test.csv', 'x')
        f.close()

    if not os.path.isfile('MachineLearning/NeuralNetwork/datasets/train.csv'):
        f = open('MachineLearning/NeuralNetwork/datasets/train.csv', 'x')
        f.close()

    for row in rows:
        random.shuffle(row)
        l = int(len(row) * 0.8)
        train, test = row[:l], row[l:]

        with open('MachineLearning/NeuralNetwork/datasets/test.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(test)
        
        with open('MachineLearning/NeuralNetwork/datasets/train.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(train)

def main(argv):
    # default values
    folder_path = ''
    whitelist = list()

    try:
        opts, args = getopt.getopt(argv,"hd:", ["whitelist="])
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
        elif opt in ("--whitelist"):
            whitelist = arg.split(',')

    if folder_path == '':
        print('Need to specify a directory')
        exit(2)

    # Check if directory exists
    if not os.path.isdir('MachineLearning/NeuralNetwork/datasets'):
            os.mkdir('MachineLearning/NeuralNetwork/datasets')

    # Check if directory exists
    if not os.path.isdir('MachineLearning/NeuralNetwork/datasets/images'):
            os.mkdir('MachineLearning/NeuralNetwork/datasets/images')

    rows = Extract(folder_path, whitelist)
    
    # Write to csv files
    WriteCSV(rows)
    #WriteCSV('MachineLearning/NeuralNetwork/datasets/train.csv', train_rows)

    print("Finished.")

if __name__ == "__main__":
   main(sys.argv[1:])

