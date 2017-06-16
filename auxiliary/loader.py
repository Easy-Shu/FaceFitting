import numpy as np
import glob
import cv2
import scipy.io

minimum_landmarks = 3

# Read a .ply file without texture
# read_faces define if it needs to read the faces
def readPLY(file_name):
	vertex_list = []
	face_list = []

	vertex_number = 0
	face_number = 0
	
	with open(file_name) as ply:
	
		# Read header
		while True:
			words = ply.readline().split()
			
			if len(words) == 0:
				continue
			
			elif words[0] == 'end_header':
				break
				
			elif len(words) == 3:
				if words[1] == 'vertex':
					vertex_number = int(words[2])
					
				elif words[1] == 'face':
					face_number = int(words[2])
					
			else: 
				continue
			
		# Read vertices
		if vertex_number > 0: 
			for x in range(vertex_number):
				line = ply.readline()
				values = [float(x) for x in line.split()]
				vertex_list.append((values[0],values[1],values[2]))						
		else: 
			raise NameError('Could not read ply file, invalid number of vertices')
					
		# Read faces
		if face_number > 0: 
			for x in range(face_number):
				line = ply.readline()
				values = line.split()
				face_list.append( (int(values[1]), int(values[2]), int(values[3])) )						
		else: 
			raise NameError('Could not read ply file, invalid number of faces')
					
				
	ply.close()
	return vertex_list, face_list		
		
# Reads file in the format described in 'share/Landmark Description.txt'. It returns a 2D list with size [num_modelLandmark, 2], where the first item in each row is the landmark index, and the second is the actual vertex index of that landmark in the model. 

# obs: subtracts one from the vertex index for 0-start indexing if set to, so the returned value must be in 0-start indexing
def readModelLandmarks(bfm_pts_path, subtractOneFromIndex=True):
	model_vertices = []
	
	with open(bfm_pts_path) as file:
		while True:
			words = file.readline().split()
			
			if len(words) == 0:
				continue
			
			elif words[0] == '#' or words[0] == '{':
				continue
				
			elif words[0] == '}':
				break
				
			elif len(words) == 2:
				landmark_index = int(words[0])
				vertex_index = int(words[1])
				
				if subtractOneFromIndex:
					vertex_index -= 1
				
				model_vertices.append([landmark_index, vertex_index])
				
			else:
				raise NameError('Unsupported file formatting encountered.')
				
	file.close()
	
	return model_vertices
	
# Reads .pts file containing list of 2D points positions and their respective landmark index. Returns a 2D list size [num_imageLandmarks, 2], where the first item of each column is the landmark index and the second is a tuple (x,y) with the landmark position in the image.
# For occluding points the returned value is actually a 1D list with all the landmark points as tuples (x,y). 
# obs: The landmark index should match the index in the BFM.pts file, unless it is a "occluding landmark", which means it is at a contour such as the facial silhouette or a nose contour. In this case the index should be 0.

def readImageLandmarks(target_pts):
	# Save here if it is a "fixed" landmark, i.e., has a corresponding position in the 3D model
	image_points_with_index = []
	
	# Save here if it is a silhouette point
	occluding_points = []
	
	with open(target_pts) as file:
		while True:
			words = file.readline().split()
			
			if len(words) == 0:
				continue
			
			elif words[0] == '#' or words[0] == '{':
				continue
				
			elif words[0] == '}':
				break
				
			elif len(words) == 3:
				
				landmark_index = int(words[0])
				point = ( float(words[1]), float(words[2]) )
				
				if landmark_index == 0:
					occluding_points.append(point)
					
				else: 
					image_points_with_index.append([landmark_index, point])
				
				
			else:
				raise NameError('Unsupported file formatting encountered.')
		
	file.close()
	
	return image_points_with_index, occluding_points
		
#	Loads a .mat file from matlab
def loadMat(bfm_path):
	mat = scipy.io.loadmat(bfm_path)
	return mat
	
# Return list of images in opencv (BGR) format 
def readImages(img_path):
	images = []
	paths = glob.glob(img_path)
	
	for path in paths:
		img = cv2.imread(path)
		if img is not None:
			images.append(img)
			
	return images
		
# DEPRECATED
# Return 2 lists, containing corresponding 3D vertices (float) and 2D image points in pixels (int). Also return the indices of the landmarks in order.
	# [in] bfm: path to BFM.pts
	# [in] face_vertex_list: list containing the vertex list
	# [in] target_pts: path to file target.pts (obs maybe change to a list to avoid reading file every time)
	# [in] vertex_index_list: list of integers with the vertices indices that should be read. They start with 1
def readLandmarks(bfm_pts_path, face_vertex_list, target_pts, vertex_index_list):
	vertex_indices = []
	landmark_vertices = []
	sel_vertices = []
	landmark_2Dpoints = []
	indices = []
	
	# Read BFM vertex indicex
	with open(bfm_pts_path) as file:
		while True:
			words = file.readline().split()
			
			if len(words) == 0:
				continue
			
			elif words[0] == '#' or words[0] == '{':
				continue
				
			elif words[0] == '}':
				break
				
			elif len(words) == 2:
				vertex_indices.append(int(words[1])-1)
				
			else:
				print('Unsupported file formatting')
		
		# Check number of landmarks read
		if len(vertex_indices) < minimum_landmarks:
			raise NameError('Minimum landmark number required not met')
		
	file.close()
		
	# Read the actual vertex positions
	# vertex_list, face_list = readPLY(avg_face)
	
	for x in vertex_indices:
		vertex_tuple = face_vertex_list[x]
		landmark_vertices.append(vertex_tuple)
		
	# Read 2D points and construct selected_vertices
	with open(target_pts) as file:
		while True:
			words = file.readline().split()
			
			if len(words) == 0:
				continue
			
			elif words[0] == '#' or words[0] == '{':
				continue
				
			elif words[0] == '}':
				break
				
			elif len(words) == 3:
				
				landmark_index = int(words[0])
				point = ( float(words[1]), float(words[2]) )
				
				# Discart landmarks with index not in vertex_index_list, for example if we want to read non-contours only.
				if landmark_index not in vertex_index_list:
					continue
				
				sel_vertices.append(landmark_vertices[landmark_index-1])
				landmark_2Dpoints.append(point)
				indices.append(landmark_index)
				
			else:
				print('Unsupported file formatting')
		
		# Check number of landmarks read
		if len(sel_vertices) != len(landmark_2Dpoints):
			raise NameError('Something went wrong')
		
	file.close()
	
	return landmark_2Dpoints, sel_vertices, vertex_indices, indices










	





	
	
	
	
		


	