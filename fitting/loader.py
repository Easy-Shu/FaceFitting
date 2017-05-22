import numpy as np

minimum_landmarks = 5

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
		

# Return 2 lists, containing corresponding 3D vertices (float) and 2D image points in pixels (int). Also return the indices of the landmarks in order.
# [in] bfm: path to BFM.pts
# [in] face_vertex_list: list containing the vertex list
# [in] target_pts: path to file target.pts (obs maybe change to a list to avoid reading file every time)
# obs: discarts contours
def readLandmarks(bfm, face_vertex_list, target_pts):
	vertex_indices = []
	landmark_vertices = []
	sel_vertices = []
	landmark_2Dpoints = []
	indices = []
	
	# Read BFM vertex indicex
	with open(bfm) as file:
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
				
				# Discart contours for now
				if (landmark_index>=1 and landmark_index<=8) or (landmark_index>=10 and landmark_index<=17):
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
	
	return landmark_2Dpoints, sel_vertices, indices











	





	
	
	
	
		


	