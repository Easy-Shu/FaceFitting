import numpy as np
import cv2
from matplotlib import pyplot as plt

keypoint_file = 'target.pts'
image_file = 'target.png'
points = 68
data = []

# Read points:
with open(keypoint_file) as file:
	line = file.readline()
	print(line)
	line = file.readline()
	print(line)
	line = file.readline()
	print(line)
	
	for x in range(points):
		line = file.readline()
		data.append([float(x) for x in line.split()])
		print(line)
		
	line = file.readline()
	print(line)

file.close()

# Load and display image_0010
img = cv2.imread(image_file)

# Draw keypoints
font = cv2.FONT_HERSHEY_SIMPLEX
for idx, point in enumerate(data):
	x, y = point
	coord = (int(x), int(y))
	cv2.circle(img, coord, 1,(0,0,255), 1)
	cv2.putText(img, str(idx+1), coord, font, 0.3, (0,0,255))
	cv2.imshow('imagem...', img)
	cv2.waitKey(0)


cv2.destroyAllWindows()

