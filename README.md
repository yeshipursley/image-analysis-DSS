# Image Analysis DSS
The files in this project is made mainly for Windows and the following steps are for Windows OS primarily. The files can however be run on other operative systems as well.

## Run Files
### Install Python and Anaconda

To run our files using our repository you need to have python: https://www.python.org/downloads/ installed on your Windows computer.
I would also recommend you download Anaconda: https://www.anaconda.com/products/individual to make it easier to run the files.

### Clone Repository
Clone repository in a wanted location on your computer using: 
```
git clone https://github.com/yeshipursley/image-analysis-DSS.git
```
### Open Anaconda Prompt
Once you have installed Anaconda you should be able to press the Windows button and search for Anaconda Prompt: 

![](images/anaconda.jpg "Anaconda Prompt")

Once you have opened that you should navigate to the repo. You can do this by using: 
```
cd <full_path to the cloned repo>
```
### Use Pip Install
Before installing the pip packages, create a virtual enviroment to make managing the pacakges easier with:
```
python3 -m venv dss
```
Then, to activate the virtual enviroment do:
```
dss\Scripts\activate
```
Terminal should now show (dss) before the prompt.

When that is done you should use: 
```
pip install -r requirements.txt
```
to install all the dependencies needed to run the files. 

### Run Image Enhancement Files
The image enhancement part of the project has been divided into three folders: Histogram, MorpholigicalTransformations and NoiseReduction. 
The way you run the python files in these three folders is the same for every folder. You should run them in the Anaconda Prompt after you have installed the
necessary dependencies using the earlier step. This is how you run them:

```
cd <folder_name>
```
Use cd to move to wanted folder.

```
python .\filename.py -input .\inputImage.jpg -output .\outputImage.jpg
```
The *.\filename.py* is the name of the python file you want to run. The *.\inputImage.jpg* is the path of the image to want to do image enhancment on. 
The *.\outputImage.jpg* is the path where you want to store the now newly created image that has been enhanced. Remember to include the filename in this path. The paths can be absolute paths or relative paths to the cloned repo.
Use backslashes ("\\") in the paths and if you use relative paths use a dot before the path (.\path\imageName.jpg).<br/>
IMPORTANT: For the adaptive_histogram.py file the image should be rgb or gray-scale. For all the other files the image should be binarized.   


### Image Segmentation Files Structure
Our image segmentation work has been divided into 5 folders:
- binarization_comparison: this file was used to test different binarization methods. 
- custom_traineddata_file: contains our custom traineddata file
- pytesseract_image_to_boxes_comparison: contains a file that was used to test how different binarization methods affect pytesseract's segmentation.
- segmentation_to_classifier: contains files needed to run our image segmentation.
- yolo_to_img: a file we created to manually convert letters extracted from LabelImg in the YOLO format to images.

### Run Image Segmentation Files
Steps to test our image segmentation:
- Copy the segmentation_to_classifier folder into another folder. 
- Download tesseract - https://github.com/UB-Mannheim/tesseract/wiki
- Copy the heb.traineddata file from the "custom_traineddata_file" folder in this GitHub repository and put it into your Tesseract-OCR/tessdata folder.
- Copy the default.model from "image-analysis-DSS/machine_learning/neural_network/models/default/" folder in this GitHub repository and paste it in your segmentation_to_classifier folder.
- In segmentation_to_classifier.py make sure the path, located at line 322, goes to your tesseract.exe file.
- In segmentation_to_classifier.py add these lines at the bottom of the code: 
```
img = cv2.imread('path to your image')

Segmentor().segmentClearBackground(img)
# or
Segmentor().segmentVariedBackground(img)
```
- Run the segmentation_to_classifier.py file. 

### Run Machine Learning Files
#### Training
```
python MachineLearning\NeuralNetwork\train.py
```
Add `--gpu` to run the training on an available GPU  
Add `-e :number:` to run the training for a set number of epochs, where `:number:` is your desired epoch (default is 20)  
Add `-d :name:` or `--dataset :name:` to run a dataset other than the default "default", datasets must be located in the `MachineLearning/NeuralNetwork/datasets` folder  
Add `-m :name:` or `--model :name:` to give the trained model a name other than "default", trained models will be saved in the `MachineLearning/NeuralNetwork/models` folder  
Add `--earlystop :loss:` to use the callback function and stop the training at a certain loss value, or until all epochs are completed  

#### Extraction
```
python MachineLearning\extract.py
```
Add `-d :path:` to specify what `:path:` folder the dataset should be extracted from, folder needs to have subfolders with all the letters with their respective name  
Add `-n :name:` or `--name :name: `to specify what `:name:` the dataset should be called, datasets are saved in the `MachineLearning/NeuralNetwork/datasets` folder 

#### Predict
```
python MachineLearning\NeuralNetwork\predict.py
```

You will need to change the model_name variable at line 114 to the desired model located in the `MachineLearning/NeuralNetwork/models` folder 

### Test image
To test our code we have added a test image called *test.png* in the repo. The image is a paragraph from The Great Isaiah Scroll column 35, gotten from: https://archive.org/details/qumran 