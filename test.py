import os

for subdir, dirs, files in os.walk('Data'):

        print("Current subdirectory: "+ subdir + " " + len(files))