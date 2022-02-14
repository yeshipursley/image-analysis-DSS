import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys, getopt

CLASSES = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED', 'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']
def Extract(folder_path):
    # Check for directories
    if not os.path.isdir('MachineLearning/NeuralNetwork/datasets/train'):
        os.mkdir('MachineLearning/NeuralNetwork/datasets/train')

    if not os.path.isdir('MachineLearning/NeuralNetwork/datasets/test'):
        os.mkdir('MachineLearning/NeuralNetwork/datasets/test')

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

            new_image = new_image.resize((28,28), resample=Image.NEAREST)

            # Save image into either training or testing folders
            image_name = label + str(i) + ".png"
            if(i < split):
                new_image.save("NeuralNetwork/datasets/train/" + image_name)
                train_rows.append((image_name, CLASSES.index(label.upper())))
            else:
                new_image.save("NeuralNetwork/datasets/test/" + image_name)
                test_rows.append((image_name, CLASSES.index(label.upper())))
        
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
            folder_path = 'MachineLearning/NeuralNetwork/' + arg

    if folder_path == '':
        print('Need to specify a directory')
        exit(2)

    # Check if directory exists
    if not os.path.isdir('MachineLearning/NeuralNetwork/datasets'):
            os.mkdir('MachineLearning/NeuralNetwork/datasets')

    print(folder_path)
    print("Looping through characters:")
    test_rows, train_rows = Extract(folder_path)

    # Write to csv files
    print("Writing to CSV files")
    WriteCSV('MachineLearning/NeuralNetwork/datasets/test.csv', test_rows)
    WriteCSV('MachineLearning/NeuralNetwork/datasets/train.csv', train_rows)

    print("Finished extracting dataset")

if __name__ == "__main__":
   main(sys.argv[1:])

