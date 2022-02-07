import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt

for subdir, dirs, files in os.walk('characters'):
    num_files = len(files)
    
    if(num_files == 0):
        continue
    
    dir = subdir[11:]
    os.mkdir('characters_small\\' + subdir[11:])
    for i, file in enumerate(files):
        if i > 10:
            break

        copyfile(subdir + "\\" + file, 'characters_small\\' + dir + "\\" + file)