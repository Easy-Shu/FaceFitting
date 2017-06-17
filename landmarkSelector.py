# Landmark selector
# Use this to select landmarks from image and save it to file 
import cv2
import numpy as np
import glob
from os.path import basename

point = (0,0)
max_index = 70
currIndex = 1
points = []
indices = []
clicked = False

font = cv2.FONT_HERSHEY_SIMPLEX
color_unmarked = (0,0,255)
color_marked = (200,0,200)

# Estimated monitor height 
monitor_height = 1400
scale = 1

# Window fuctions
def mouse_event(event, x, y, flags, param):
	global point
	global clicked
	global currImage
	global previousImage
	global scale
	global color_unmarked
	
	if event == cv2.EVENT_LBUTTONDOWN:
		clicked = True
		unscale = 1/scale
		point = (round(x*unscale), round(y*unscale))
		print(point)
		
		#R, G, B = currImage[y, x, :]
		#bright = 0.5 * max(R, G, B) + 0.5*min(R, G, B)
		#print('brightness: %f' % bright)
		
		currImage = previousImage
		previousImage = currImage.copy()
		currImage = cv2.circle(currImage, (x, y), 2, color_unmarked, -1)

pts_extension = '.pts'

folder = input('Folder: ')
extension = input('Image extension: ')

# Load images
path = folder + '/*.' + extension
files = glob.glob(path)
names = [basename(x).split('.')[0] for x in files]

images = []

for file_name in files:
	img = cv2.imread(file_name)
	images.append(img)
	
scale = monitor_height / max([images[0].shape[0], images[0].shape[1], images[0].shape[2] ])
print(scale)

for i in range(len(images)):
	point = (0,0)
	max_index = 70
	currIndex = 1
	points = []
	indices = []
	clicked = False
	img = images[i]
	
	currImage = cv2.resize(img, (0,0), fx = scale, fy = scale)
	previousImage = np.copy(currImage)

	# Set window
	cv2.namedWindow('Select keypoints')
	cv2.setMouseCallback('Select keypoints', mouse_event)

	print('Current index: %d' % currIndex)

	while True:
		cv2.imshow('Select keypoints', currImage)
		key = cv2.waitKey(10) & 0xFF
		
		if key == 27:
			if clicked:
				clicked = False
				currImage = previousImage
				
			else:
				break
		
		elif key == 13:
		
			# Save landmark position if somewhere was clicked
			if clicked:
				x, y = point
				x = round(x * scale)
				y = round(y * scale)
				currImage = cv2.circle(currImage, (x, y), 3, color_marked, -1)
				cv2.putText(currImage, str(currIndex), (x, y), font, 0.5, (0,0,0))
				previousImage = currImage
				clicked = False
				indices.append(currIndex)
				points.append(point)
				print('Landmark Saved\n')
			
			currIndex+=1	
			if currIndex > max_index or currIndex == 1:
				currIndex = 0
				
			print('Current index: %d' % currIndex)
			
	cv2.destroyAllWindows()
	
	pts_file = folder+'/'+ names[i] +pts_extension
	file = open(pts_file,'w')
	file.write('{\n')
	
	for i in range(len(indices)):
		index = indices[i]
		x, y = points[i]
		file.write('%d %d %d\n' % (index, x, y))
	
	file.write('}\n')
	file.close()
	print('Saved %s' % pts_file)