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
When you have navigated to the cloned repo you should use: 
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
Use backslashes ("\\") in the paths and if you use relative paths use a dot before the path (.\path\imageName.jpg).   


### Run Image Segmentation Files


### Run Machine Learning Files