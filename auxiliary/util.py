import numpy as np

# Function that receives the list of model landmarks, image landmarks and landmarks indices to be considered. Returns two lists with the model vertices indices and the corresponding image point landmarks
def fixedLandmarkCorrespondence(model_vertices, image_points, landmark_indices):
	Vertices_indices = []
	Points = []

	indices1 = [i for i,j in model_vertices]
	indices2 = [i for i,j in image_points]
	
	mv_index=0
	
	for i in landmark_indices:
		
		# find if there is this index in model_vertices
		try:
			mv_index = indices1.index(i)
			
		# If not found, go to next iteration
		except ValueError:
			continue
			
		# find if there is this index in image_points
		try:
			ip_index = indices2.index(i)
			
		except ValueError:
			continue
			
		# If no exception was thrown, means we found landmark index i in both lists
		vertex = model_vertices[mv_index][1]
		point = image_points[ip_index][1]
		
		Vertices_indices.append(vertex)
		Points.append(point)
	
	return Vertices_indices, Points

# Returns list of vertices (in tuple form) given list of vertex indices and current mesh
# obs: assumes vertex_indices is in 0-start indexing
def verticesFromCurrentMesh(vertex_indices, currentMesh):
	Vertices = []

	# Reshape mesh to size [n,3] in case it is in wrong formatting
	mesh = currentMesh.reshape((-1,3))
	
	for i in vertex_indices:
		x, y, z = mesh[i,:]
		Vertices.append((x, y, z))
		
	return Vertices

# Converts list of points [(x1,y1), (x2,y2)... ] to 2D array [[x1], [y1], [1]], [[x2] ... ] ...]
def pointsTo2DArray_homogeneous(tupleList):
	flatPoints = []
	
	for tuple in tupleList:
		x, y = tuple
		flatPoints.extend([x,y,1])
		
	array1D = np.array(flatPoints)
	array2D = np.reshape(array1D, (-1,1))
	
	return array2D
	
# Converts list of points [(x1,y1), (x2,y2)... ] to 2D array [[x1], [y1], [[x2] ... ] ...]
def pointsTo2DArray(tupleList):
	flatPoints = []
	
	for tuple in tupleList:
		x, y = tuple
		flatPoints.extend([x,y])
		
	array1D = np.array(flatPoints)
	array2D = np.reshape(array1D, (-1,1))
	
	return array2D
	
# Fix y axis convention for a set of 2D points
# points is a list of tuples
def flipY(points, height):
	new_points = []
	
	for p in points:
		x,y = p
		new_points.append((x, height-y))
	
	return new_points
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	