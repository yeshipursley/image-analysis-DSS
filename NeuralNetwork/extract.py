import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys, getopt

CLASSES = classes = ['alef', 'ayin', 'bet', 'dalet', 'gimel', 'het', 'he', 'kaf', 'lamed', 'mem', 'nun', 'pe', 'qof', 'resh', 'samekh', 'shin', 'tav', 'tet', 'tsadi', 'vav', 'yod', 'zayin']

def Extract(folder_path):
    # Check for directories
    if not os.path.isdir('datasets/train'):
        os.mkdir('datasets/train')

    if not os.path.isdir('datasets/test'):
        os.mkdir('datasets/test')

    test_rows = list()
    train_rows = list()
    
    # for each subdirectory in folder
    for subdir, dirs, files in os.walk(folder_path):
        num_files = len(files)
        print("Current subdirectory: "+ subdir)
        if(num_files == 0):
            continue

        split = int(num_files * 0.8)
        label = subdir[len(folder_path)+1:-10]

        for i, file in enumerate(files):
            image_path = subdir + "\\" + file     
            ## Open image              
            image = Image.open(image_path).convert('L')  
            ## Create empty bigger image           
            new_image = Image.new(image.mode, (64,64), 255)
            # Calculate center position         
            x, y = 32 - int(image.width/2), 32 - int(image.height/2)          
            # Paste image in the middle of the emtpy image
            new_image.paste(image, (x,y))

            #new_image = image.resize((32,32))

            # Save image into either training or testing folders
            image_name = label + str(i) + ".png"
            if(i < split):
                new_image.save("datasets/train/" + image_name)
                train_rows.append((image_name, CLASSES.index(label)))
            else:
                new_image.save("datasets/test/" + image_name)
                test_rows.append((image_name, CLASSES.index(label)))
        
    return (test_rows, train_rows)

def WriteCSV(filepath, rows):
    if not os.path.isfile(filepath):
        f = open(filepath, 'x')
        f.close()

    # Write to csv files
    with open(filepath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

def main(argv):
    # default values
    folder_path = ''

    try:
        opts, args = getopt.getopt(argv,"hd:", ["directory="])
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

    if folder_path == '':
        print('Need to specify a directory')
        exit(2)

    # Check if directory exists
    if not os.path.isdir('datasets'):
            os.mkdir('datasets')

    print(folder_path)
    print("Looping through characters:")
    test_rows, train_rows = Extract(folder_path)

    # Write to csv files
    print("Writing to CSV files")
    WriteCSV('datasets/test.csv', test_rows)
    WriteCSV('datasets/train.csv', train_rows)

    print("Finished extracting dataset")

if __name__ == "__main__":
   main(sys.argv[1:])

