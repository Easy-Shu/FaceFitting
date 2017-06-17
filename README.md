# Face Fitting

Little project in python based on project from Patrick Hubber (https://github.com/patrikhuber/eos). Idea is to fit a morphable model shape (the basel face model) to face image, given keypoints in the image and their correspondences in the model.

## Contents 

* auxiliary : Here are the main functions that do most of the job, including camera estimation, edge correspondences finder, loading functions, texture extraction, debugging functions (in draw.py)
* Classes : I was not going to use classes at the beginning but project management became impossible so I implemented them. They basically handle the auxiliary functions usage nicely. 
* Landmark Manual : Explaining how to chose the landmark locations etc. _annotated.ply_ has all the fixed landmark positions. 
* out : I usually output everything here and then move to another place. 
* share : Has every file necessary for execution, except for target specific files like target image and target landmarks. Some files could be thrown away, specially the .obj and .mtl files, but I use them as template to output obj file with texture. 
* target : Usually has a target specific data. This means images of the same target (from different angles if more than one) and files containing the landmark locations. 
* demo.py : Usage demo includng every feature
* landmarkSelector.py : A useful script I made that helps selecting landmark locations

## Requirements

* Python 3.5, with some standard libraries (numpy, opencv 3, scipy, ...)
* Basel Face Model (the .mat file) 
* Image with keypoint file. I will provide one. 
