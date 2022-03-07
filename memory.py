import os, sys
from PIL import Image
images = list()
for subdir, dirs, files in os.walk('Data/tabita_dataset'):
        num_files = len(files)
        
        if(num_files == 0):
            continue

        print("Current subdirectory: "+ subdir)
        for i, file in enumerate(files):
            image = Image.open(subdir + '\\' + file)
            images.append(image)

print(len(images))
print(sys.getsizeof(images))