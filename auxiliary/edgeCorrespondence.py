import auxiliary.orthogonalCamera as camera

import numpy as np
import scipy.spatial as spatial
import cv2

# Returns correspondences between all the contour/occluding vertices and image 2D occluding points, given the current mesh and camera params for single image
#	[in] currentMesh: array size [n, 3] with the vertices of the current face mesh in each row
#	[in] R, t, s: Current camera parameters
#	[in] imageContourPoints: The face-occluding points in the image, i.e., contour/silhouette defining points.
#	[in] occludingVertices_i: list of vertices indices (hence _i, for index) that are occluding in the given Rotation. Maybe this should be calculated inside of the function, but then it would require too many arguments.
# 	[in] min_dist: Radious for finding closest contour points. Should be as small as possible for more accurate results, and will vary based on the size of image used. For now it will be determined experimentally

# obs: occludingVertices_i should match the current Rotation! 
def contourCorrespondences(currentMesh, R, t, s, imageContourPoints, occludingVertices_i, min_dist=15):
	vertex_indices = []
	points = []
	
	# Guarantees currentMesh is in right shape
	V = currentMesh.reshape(-1,3)

	# Get all projected occluding vertices and their respective Z component after rotation (for Z-buffering approximation)
	projectedOccudingVertices = []
	verticesDepth = []
	
	for v_index in occludingVertices_i:
	
		# Get vertex given current shape
		vertex = V[v_index, :]
		
		# Get vertex projection (1D array size [2,] )and save
		projected = camera.SOP(vertex, R, t, s)
		projectedOccudingVertices.append(projected)
		
	# e.g., index of highest Z vertex value:
	# highest_i = occludingVertices_i[verticesDepth.index(max(verticesDepth))]
	
	# Build cKDTree for finding closest points
	projected_points = np.array(projectedOccudingVertices)
	p_tree = spatial.cKDTree(projected_points)
	
		
	# Now find contour correspondences - this bit of code is a little complicated, but it is basically trying to find the vertices that project the closest to a landmark occluding point given the camera parameters, and then pick the one with the highest Z component
		
	for im_point in imageContourPoints:
		# 1. Find all projected occluding vertices within 'min_dist' distance from landmark point
		nearest_i = p_tree.query_ball_point(im_point, min_dist)
		
		if len(nearest_i) == 0:
			print('No occluding vertex projected was found near landmark point at %s, so it will be ignored in the fitting.' % str(im_point))
			print('Increasing contour tolerance might help this, but it might also be possible that the landmark location does not have any occluding vertex. In this case, increasing min_dist will cause a fitting problem. \n')
			
			continue
		
		# Find closest projection
		dist = 100000
		for v_index in nearest_i:
			px, py = projected_points[v_index, :]
			x, y = im_point
			d2 = np.power((px-x),2) + np.power((py-y),2)
			if d2 < dist:
				vertex_index_nearest = occludingVertices_i[v_index]
				dist = d2
				
		# Save it to result
		points.append(im_point)
		vertex_indices.append(vertex_index_nearest)
		
	
	return vertex_indices, points
	
# Same as occludingBoundaryVertices, except it looks for faces whose normals have Z component bigger than 0 and smaller than a threshold (maxZ)
def silhouetteVertices(currentMesh, meshFaces, R, visibleOnly = True):
	# face normals with Z values bellow minZ will be considered silhouette
	maxZ = 0.13
	silhouettes = []
	# this value is for basel face model and will vary from model to model. 
	# It is the depth of the face at a point near the neck. We consider that silhouettes won't have a depth value lower than this, so we avoid many visible points that aren't silhouettes. 
	threshold = 46000.0
	
	# reshape to [n x 3], so each vertex is in each row
	V = np.copy(currentMesh.reshape(-1,3))
	F = np.copy(meshFaces.reshape(-1,3))
	
	RotV = np.copy(V)
	
	# transform each vertex by rotation matrix
	for i in range(V.shape[0]):
		vertex = V[i, :]
		RotV[i, :] = np.matmul(R, vertex.reshape((3,1))).flatten()
		
	# compute face normals
	fn = faceNormals(RotV, F)
	
	# Test if silhouette for each triangle face
	for i in range(F.shape[0]):
		# fn[i] corresponds to F[i,:], so test fn[i]
		
		normalZ = fn[i, 2]
		
		if normalZ >= 0 and normalZ <= maxZ:
			v1, v2, v3 = F[i, :]
			v1 -= 1
			v2 -= 1
			v3 -= 1
			
			if V[v1, 2] < threshold or V[v2, 2] < threshold or V[v3, 2] < threshold:
				continue
			
			silhouettes.extend([v1, v2, v3])
			
	silhouettes = list(set(list(silhouettes)))
	result = []
	
	print('total contours: %d' % len(silhouettes))

	# remove invisible vertices
	if visibleOnly:
		visible = visibleVertices(V, meshFaces, R)
		
		for i in silhouettes:
			if i in visible:
				result.append(i)

		print('visible only contours: %d\n' % len(result))
	
	return result
	

# From 3DMM-edges https://github.com/waps101/3DMM_edges 
# Returns list of vertices lying on contour edges, i.e., those which the visibility of adjacent triangle faces change
# 	[in] currentMesh: vertices of current face shape (i.e., not necessarily the mean shape)
#	[in] meshFaces: also known as triangle list, this is the same throughout the whole computation
#	[in] Ef: num(edges) by 2 matrix storing faces adjacent to each edge
#	[in] Ev: num(edges) by 2 matrix storing vertices adjacent to each edge
#	[in] R: 3x3 rotation matrix
# obs: currentMesh is in 2D ndarray form, [3n x 1] or [n x 3]
def occludingBoundaryVertices(currentMesh, meshFaces, Ef, Ev, R):
	# Z values with module bellow minZ will be considered 0
	minZ = 0.01

	# reshape to [n x 3], so each vertex is in each row
	V = np.copy(currentMesh.reshape(-1,3))
	F = np.copy(meshFaces.reshape(-1,3))
	
	# transform each vertex by rotation matrix
	for i in range(V.shape[0]):
		vertex = V[i, :]
		V[i, :] = np.matmul(R, vertex.reshape((3,1))).flatten()
		
	# compute face normals
	fn = faceNormals(V, F)
	
	# maybe ignore this...
	# compute invalid indices of Ef (those edges which there is only one adjacent face = boundary edges)
	# invalid_Ef_indices = [i for i in range(Ef.shape[0]) if Ef[i, 0] == 0 ]
	
	# find all ooccludings edge indices = those which the Z sign of their adjacent face's normal change
	edge_indices = []
	for i in range(Ef.shape[0]):
		face1, face2 = Ef[i, :]
		
		# ignore boundaries
		if face1 == 0:
			continue
			
		# 0-start indexing...
		face1 -= 1
		face2 -= 2
		
		# get z component of normal of each face
		z1 = fn[face1, 2]
		z2 = fn[face2, 2]
		
		if np.absolute(z1) < minZ :
			z1 = 0
		if np.absolute(z2) < minZ :
			z2 = 0
		
		if np.sign(z1) != np.sign(z2):
			edge_indices.append(i)
			
	# select vertices from edge list above
	contourVertices = []
	for i in edge_indices:
		v1, v2 = Ev[i, :]
		contourVertices.append(v1-1)
		contourVertices.append(v2-1)
		
	# remove duplicates
	contourVertices = list(set(contourVertices) )
	
	# remove invisible vertices
	visibleVertices = removeInvisibleVertices(V, F, contourVertices)
	
	# Here we will remove the invisible vertices. 
	# The idea is to rasterize each mesh triangle into a map with size equal to the image and assign a depth 
	
	print('total contours: %d' % len(contourVertices))
	print('visible only contours: %d' % len(visibleVertices))
	
	return visibleVertices
	
# Return set of indidices of visible vertices of model, given rotation
# returnNormals: I added this term to return the normals of the rotated face. This makes computations faster in some places that normals are required so we don't calculate twice.
# Obs: this is an approximation, it assumes visible vertices have Z component of face normal positive.
# Works well for convex-like surfaces and those that don't have occluding surfaces very close to each other. 
def visibleVertices(mesh, F, R, returnNormals = False):
	minZ = 0.2
	V = np.copy(mesh.reshape((-1,3)))
	visible_vertices = []
	visible_faces = []
	
	# Rotate shape
	for i in range(V.shape[0]):
		vertex = V[i, :].reshape((3,1))
		V[i, :] = np.matmul(R, vertex).flatten()
		
	# Calculate normals
	fn = faceNormals(V, F)
	
	# Iterate through faces and find all that have normal Z component bigger than threshold
	for i in range(F.shape[0]):
		if fn[i, 2] < minZ:
			continue
		
		index_v1 = int(F[i, 0]) -1
		index_v2 = int(F[i, 1]) -1
		index_v3 = int(F[i, 2]) -1
		
		visible_vertices.extend([index_v1, index_v2, index_v3])
		visible_faces.append(i)
	
	visible_vertices = list(set(visible_vertices))
	
	# Filter those that fall behind a face 
	map_size = 2000
	rasterMap = np.full((map_size, map_size), -1000000.0, np.float32)
	
	# Project vertices
	UV = np.copy(V[:, 0:2])
	
	# Transform to image coordinates (min UV -> 0, max UV -> map_size)
	Xaxis = UV[:,0]
	Yaxis = UV[:,1]
	minX = min(Xaxis.flatten().tolist())
	minY = min(Yaxis.flatten().tolist())
	
	for i in range(Xaxis.shape[0]):
		Xaxis[i] = Xaxis[i] - minX
		Yaxis[i] = Yaxis[i] - minY
		
	maxVal = max(UV.flatten().tolist())
	
	for x in np.nditer(UV, op_flags=['readwrite']):
		x[...] = x/(maxVal+1) * map_size
		
	triangles = []
	triangle_bounds = []
	weights = []
		
	# Get each triangle projection and bounds
	for i in visible_faces:
		index_v1 = int(F[i, 0]) -1
		index_v2 = int(F[i, 1]) -1
		index_v3 = int(F[i, 2]) -1
		
		# weight = [w1, w2, w3]
		weight = [V[index_v1, 2], V[index_v2, 2], V[index_v3, 2]]
		weights.append(weight)
		
		tri = np.zeros((3, 2))
		# tri = [[x,y],
		#		 [x,y],
		#		 [x,y]]
		
		tri[0, :] = UV[index_v1, :]
		tri[1, :] = UV[index_v2, :]
		tri[2, :] = UV[index_v3, :]
		
		# Find triangle bounds
		Xs = tri[:, 0].tolist()
		Ys = tri[:, 1].tolist()
		
		minx = int(min(Xs))
		miny = int(min(Ys))
		maxx = int(max(Xs))
		maxy = int(max(Ys))
		
		# Constrain bounds to map limits
		bound = [max(0, minx), min(map_size-1, maxx), max(0, miny), min(map_size-1, maxy)]
		
		triangles.append(tri)
		triangle_bounds.append(bound)
		
	# Compute the depth of each point in the map
	for i in range(len(triangles)):
		w1, w2, w3 = weights[i]	
		tri = triangles[i]
		
		# This is approx. Depth is the average depth of the three points
		avg_depth = (w1 + w2 + w3 )/3
		
		# Test for pixels inside bounds 
		bound = triangle_bounds[i]
		minx = bound[0]
		maxx = bound[1]
		miny = bound[2]
		maxy = bound[3]
		
		if minx > maxx or miny > maxy:
			continue
			
		# for each pixel inside bounds then save depth to map if it is higher than current depth value
		for x in range(minx, maxx-1):
			for y in range(miny+1, maxy):
				
				if rasterMap[x, y] < avg_depth:
					rasterMap[x, y] = avg_depth
					
	
	tolerance = 2000
	result = []
	for v_index in visible_vertices:
		x, y = UV[v_index, :]
		depth = V[v_index, 2]
		if depth >= (rasterMap[int(x), int(y)] - tolerance):
			result.append(v_index)
				
	if returnNormals:
		return set(result), fn
		
	else :
		return set(result)
	
# Deprecated, it doesn't really work well, I dont really know why and is too computationally consuming
# Use visibleVertices() instead.
# Removes the invisible vertices from list 
def removeInvisibleVertices(V, F, contourVertices, R):
	# Here we will remove the invisible vertices. 
	# The idea is to rasterize each mesh triangle into a map with a size we specify. The bigger the map the more precise this will be, but longer it will take.
	
	# Defining map size. We'll use a square map
	map_size = 1000
	
	# Each position on map has the Z value of each rastered pixel. 
	rasterMap = np.full((map_size, map_size), -1000000.0, np.float32)
	
	# Project vertices
	UV = np.copy(V[:, 0:2])
	
	# Transform to image coordinates (min UV -> 0, max UV -> map_size)
	Xaxis = UV[:,0]
	Yaxis = UV[:,1]
	
	minX = min(Xaxis.flatten().tolist())
	minY = min(Yaxis.flatten().tolist())
	
	for i in range(Xaxis.shape[0]):
		Xaxis[i] = Xaxis[i] - minX
		Yaxis[i] = Yaxis[i] - minY
		
	maxVal = max(UV.flatten().tolist())
	
	for x in np.nditer(UV, op_flags=['readwrite']):
		x[...] = x/(maxVal+1) * map_size
		
	triangles = []
	triangle_bounds = []
	weights = []
	
	# Get each triangle projection and bounds
	for i in range(F.shape[0]):
		index_v1 = int(F[i, 0]) -1
		index_v2 = int(F[i, 1]) -1
		index_v3 = int(F[i, 2]) -1
		
		# weight = [w1, w2, w3]
		weight = [V[index_v1, 2], V[index_v2, 2], V[index_v3, 2]]
		weights.append(weight)
		
		tri = np.zeros((3, 2))
		# tri = [[x,y],
		#		 [x,y],
		#		 [x,y]]
		
		tri[0, :] = UV[index_v1, :]
		tri[1, :] = UV[index_v2, :]
		tri[2, :] = UV[index_v3, :]
		
		# Find triangle bounds
		Xs = tri[:, 0].tolist()
		Ys = tri[:, 1].tolist()
		
		minx = int(min(Xs))
		miny = int(min(Ys))
		
		maxx = int(max(Xs))
		maxy = int(max(Ys))
		
		# bound = [minx, maxx, miny, maxy]
		# First we need to constrain the bounds to the map limits
		
		bound = [max(0, minx), min(map_size-1, maxx), max(0, miny), min(map_size-1, maxy)]
		
		triangles.append(tri)
		triangle_bounds.append(bound)
		
	# Compute the depth of each point in the map 
	for i in range(len(triangles)):
		
		w1, w2, w3 = weights[i]	
		tri = triangles[i]
		
		# Test for pixels inside bounds 
		bound = triangle_bounds[i]
		minx = bound[0]
		maxx = bound[1]
		miny = bound[2]
		maxy = bound[3]
		
		if minx > maxx or miny > maxy:
			continue
			
		
		# for each pixel inside bounds, test if it is inside triangle and then compute its depth and save to map if it is higher than current depth value
		for x in range(minx, maxx+1):
			for y in range(miny, maxy+1):
				
				# Compute barycentric coordinates of pixel
				
				# NORMAL METHOD
				####
				v, w, u = barycentricCoords(float(x), float(y), tri)
				# To test if inside triangle make: inside iff 0 < v < 1; 0 < w < 1; 0 < u < 1
				insideTriangle = (v >= -0.01 and v <= 1.01 and u >= -0.01 and u <= 1.01 and w >= -0.01 and w <= 1.01)
				if not insideTriangle:	
					continue
				
				# Compute depth 
				depth = w1 * v + w2 * w + w3 * u
				
				# Save to map if higher than current depth in map
				if rasterMap[x, y] < depth:
					rasterMap[x, y] = depth
				###
				
				# APPROXIMATION METHOD
				#depth = w1 * 0.33 + w2 * 0.33 + w3 * 0.33
				#if rasterMap[x, y] < depth:
				#	rasterMap[x, y] = depth
				
					
	# Test visibility of each vertex in contourVertices
	visibleVertices = []
	tolerance = 15
	
	for v_index in contourVertices:
		x, y = UV[v_index, :]
		
		depth = V[v_index, 2]
		
		if depth >= (rasterMap[int(x), int(y)] - tolerance):
			visibleVertices.append(v_index)
		
	return visibleVertices
				
		
# Determine whether 2D point lie inside triangle
# Taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
# Triangle is 2D array size [3,2]
def pointInTriangle(px, py, triangle):
	p0x, p0y = triangle[0, :]
	p1x, p1y = triangle[1, :]
	p2x, p2y = triangle[2, :]
	
	A = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
	sign = 1
	
	if A < 0:
		sign *= -1
		
	s = (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py) * sign
	t = (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py) * sign
	
	return s > 0 and t > 0 and (s + t) < 2 * A * sign
	
def barycentricCoords(px, py, triangle):
	p0x, p0y = triangle[0, :]
	p1x, p1y = triangle[1, :]
	p2x, p2y = triangle[2, :]
	
	v0x = p1x - p0x
	v0y = p1y - p0y
	v1x = p2x - p0x
	v1y = p2y - p0y
	v2x = px - p0x
	v2y = py - p0y
	
	den = v0x * v1y - v1x * v0y
	v = (v2x * v1y - v1x * v2y) / den
	w = (v0x * v2y - v2x * v0y) / den
	u = 1.0 - v - w
	
	return v, w, u

# Returns an array with size [num_triangles, 3], containing the normal vector of each triangle face in the mesh
def faceNormals(meshVertices, meshFaces):
	V = meshVertices.reshape(-1,3)
	F = meshFaces.reshape(-1,3)
	
	num_faces = F.shape[0]
	
	normals = np.zeros((num_faces, 3))
	
	for i in range(num_faces):
	
		# Get the index of each vertex for each triangle (face)
		# Does subtract 1 for 0-start indexing (this indexing convention is killing me)
		index_v1 = int(F[i, 0]) -1
		index_v2 = int(F[i, 1]) -1
		index_v3 = int(F[i, 2]) -1
		
		v1 = V[index_v1, :] 
		v2 = V[index_v2, :] 
		v3 = V[index_v3, :] 
		
		e1 = v2 - v1
		e2 = v3 - v1
		
		# cross product of edges gives the normal face
		fn = np.cross(e1, e2)
		
		# normalize vector
		norm = np.linalg.norm(fn)
		fn = fn / norm
		
		# We would be done here, however it turns out the face indexing in the Faces archive has the orders swapped, so it ends up calculating the normals pointing to the inside of the face instead of the outside. To fix this we will change the sign of each component
		#fn = -fn  # not anymore, already swapped cols in FaceModel loading stage
		
		normals[i, :] = fn
			
	
	return normals
	
# Returns the Yaw angle from rotation matrix in degrees
def yawAngle(R):
	angles, jac = cv2.Rodrigues(R)
	yaw = np.rad2deg(angles)[1,0]
	return yaw
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		