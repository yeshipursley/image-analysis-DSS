import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HebrewDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Create a CSV file with all the resized images
# path = 'datasets/images/'
# num_files = len(os.listdir(path))


# dataset = list()
# for i, file in enumerate(os.listdir(path)):
#     label = re.sub(r'\d+', '', file)[:-4] # Strip the name from the filepath
#     image = Image.open(path+file).convert('L') # Open the image
#     dataset.append([file, label])
   
# # Write to csv
# f = open('datasets/dataset.csv', 'w')
# writer = csv.writer(f)
# writer.writerows(dataset)
# f.close()
    
