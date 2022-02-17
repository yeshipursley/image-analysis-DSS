import os

for subdir, dirs, files in os.walk('raw_data/merged_dataset'):
    print(subdir[len('raw_data/merged_dataset/'):] + " has " + str(len(files)))