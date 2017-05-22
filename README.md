# Face Fitting

Little project in python based on project from Patrick Hubber (https://github.com/patrikhuber/eos). Idea is to fit a morphable model (the basel face model) to face image, given keypoints in the image and in the face model. The project is still in its very beginning. 

## Contents 

* avg_face.ply: Average face used to map the landmarks and corresponds to the average face used in the program
* target.png: the target image, whose face the model will be fit to
* target.pts: list containing the 2D pixel location of the landmarks, using a left-to-right up-to-bottom coordinate convention. (top left pixel is 0,0)
* .py files: the source code

## Requirements

* Python 3.5, with some standard libraries (numpy, opencv 3...)
* Basel Face Model (possibly the .mat file)
* Image with keypoint file. I will provide one. 
