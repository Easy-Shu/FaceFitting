import numpy as np
import cv2

def drawKeypoints(image, points2D):
	img = np.copy(image)
	
	for point in points2D:
		x, y = point
		coord = (int(x), int(y))
		cv2.circle(img, coord, 1,(0,0,255), 1)
		
	return img
	
def drawKeypointsWithIndices(image, points2D, indices):
	img = np.copy(image)
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i, point in enumerate(points2D):
		x, y = point
		coord = (int(x), int(y))
		cv2.circle(img, coord, 1,(0,0,255), 1)
		cv2.putText(img, str(indices[i]), coord, font, 0.5, (0,0,255))
		
	return img