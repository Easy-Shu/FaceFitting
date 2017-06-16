import numpy as np
import auxiliary.orthogonalCamera as camera
import cv2

# Draws the vertices whose indices are in vertexIndexList onto image using given camera paramenters and given mesh shape
def drawVertices(image, vertexIndexList, meshShape, R, t, s, yFlip=True, size = 1.0):
	
	points = []
	height = image.shape[0]
	
	meshShape.shape = (-1, 3)
	for vindex in vertexIndexList:
		v = meshShape[vindex, :]
		point = camera.SOP(v, R, t, s)		
		points.append(point)
		
	img = drawKeypoints(image, points, yFlip, color = (0,255,0), size = size)
	
	return img

def drawKeypoints(image, points2D, yFlip=True, color = (0,0,255), size = 1.0):
	img = np.copy(image)
	height = img.shape[0]
	
	for point in points2D:
		x, y = point
		
		if yFlip:
			y = height-y
		
		coord = (int(x), int(y))
		cv2.circle(img, coord, int(1/size), color, int(1/size))
		
	# Resize image if set to
	if size != 1.0:
		img = cv2.resize(img, (0,0), fx=size, fy=size) 
		
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
	
#	Write PLY file shape only, considering the rotation matrix given
def writeRotatedPLY(file_path, V, F, R):

	# Copy, otherwise it will rotate original
	Vertices = np.copy(V.reshape((-1, 3)))
	
	for i in range(Vertices.shape[0]):
		vertex = Vertices[i, :]
		Vertices[i, :] = np.matmul(R, vertex)
	
	# Write using writeAnnotatedPLY without any annotations
	writeAnnotatedPLY(file_path, Vertices, F, [])
		

#	Writes PLY with annotated vertices contained in list vertex_list
#	vertex_list assumes 0-start indexing
def writeAnnotatedPLY(file_path, V, F, vertex_list):
	Vertices = V.reshape((-1, 3))
	Faces = F.reshape((-1,3)).astype(int)
	
	num_vertices = Vertices.shape[0]
	num_faces = Faces.shape[0]
	
	#	All white, except for the annotated vertices
	texture = np.full((num_vertices, 3), 255, np.int32)
	
	for i in vertex_list:
		texture[i, :] = np.array([0,0,0])
		
	#	Write header
	f = open(file_path, 'w')
	f.write('ply\n')
	f.write('format ascii 1.0\n')
	f.write('comment VCGLIB generated\n')
	f.write('element vertex '+ str(num_vertices) +'\n')
	f.write('property float x\n')
	f.write('property float y\n')
	f.write('property float z\n')
	f.write('property uchar red\n')
	f.write('property uchar green\n')
	f.write('property uchar blue\n')
	f.write('element face '+ str(num_faces) +'\n')
	f.write('property list uchar int vertex_indices\n')
	f.write('end_header\n')
	
	#	Write vertices
	for i in range(num_vertices):
		x, y, z = Vertices[i, :]
		r, g, b = texture[i, :].astype(int)
		
		f.write(str(x)+' '+str(y)+' '+str(z)+ ' '+ str(r)+' '+str(g)+' '+str(b)+' \n')
		
	#	Write faces
	for i in range(num_faces):
		f1, f2, f3 = Faces[i, :]
		
		#	Swapping f2 with f1. Another convention. It has to be clockwise
		#	Also subtracting 1 because PLY use 0-start indexing
		f.write('3 '+ str(f1-1)+' '+str(f2-1)+' '+str(f3-1)+' \n')
	
	f.close()