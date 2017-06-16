from Classes.pose import Pose

import auxiliary.orthogonalCamera as camera 
import auxiliary.edgeCorrespondence as edges
import auxiliary.draw as draw

import numpy as np
import cv2


# Reads the base texture file in opencv format
# Returns image in format [height, width, channel: BGR]
def readTexture(file_name):
	img = cv2.imread(file_name, cv2.IMREAD_COLOR )
	return img
	
# Extracts texture from image given the pose and the model
# obs: assumes FV and FT ate in 1-starting indexing
def extractTextureSinglePose(base_texture, pose, estimated_shape, tex_points, FV, FT):
	V = estimated_shape.reshape((-1,3))

	tex_dst = np.copy(base_texture)
	texture_height = tex_dst.shape[0]
	texture_width = tex_dst.shape[1]
	
	pose_image = pose.image
	image_height = pose_image.shape[0]
	R = pose.R
	t = pose.t
	s = pose.s
	
	mask = np.zeros((texture_height, texture_width))
	
	num_faces = FV.shape[0]
	num_vertices = V.shape[0]
	
	# Calculate visible only vertices (takes a while because we're basically rendering the whole mesh using CPU)
	visible_vertex_i = edges.visibleVertices(V, FV, R)
		
	for i in range(num_faces):
	
		# Calculate projection of each face vertex into image
		vi1 = FV[i, 0] - 1
		vi2 = FV[i, 1] - 1
		vi3 = FV[i, 2] - 1
		
		# Test for visibility
		visible = 0
		if vi1 in visible_vertex_i:
			visible+= 1
			
		if vi2 in visible_vertex_i:
			visible+= 1
			
		if vi3 in visible_vertex_i:
			visible+= 1
		
		# Test if at least 1 are visible 
		if visible < 1: 
			continue
		
		v1 = V[vi1, :]
		v2 = V[vi2, :]
		v3 = V[vi3, :]
		
		p1 = camera.SOP(v1, R, t, s)
		p2 = camera.SOP(v2, R, t, s)
		p3 = camera.SOP(v3, R, t, s)
		
		# Get texture corresponding triangle
		# Here we're assuming texture height equals texture width for simplicity, but wouldn't be hard to take both into account
		ti1 = FT[i, 0] - 1
		ti2 = FT[i, 1] - 1
		ti3 = FT[i, 2] - 1

		t1 = tex_points[ti1, :] * texture_height
		t2 = tex_points[ti2, :] * texture_height
		t3 = tex_points[ti3, :] * texture_height
		
		# We ahve 2 problems concerning axis convention here: 
		# 1. Images loaded by opencv (includes every image file here) use [y ,x] convention. 
		# However, warp affine etc use normal x,y convention 
		# 2. opencv points are in y-upwards convention. Ours are in y-downwards, including the texture coords. 
		# So we will subtract the height from y here 
		
		p1[1] = image_height - p1[1]
		p2[1] = image_height - p2[1]
		p3[1] = image_height - p3[1]
		
		t1[1] = texture_height - t1[1]
		t2[1] = texture_height - t2[1]
		t3[1] = texture_height - t3[1]
		
		# This bit of code was based on https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
		
		# Define source and destination triangles (respec. 1 & 2)
		tri1 = np.vstack((p1, p2, p3)).astype(np.float32)
		tri2 = np.vstack((t1, t2, t3)).astype(np.float32)
		
		# Define bounding boxes
		r1 = cv2.boundingRect(tri1)
		r2 = cv2.boundingRect(tri2)
		
		# Crop scr image and triangles
		tri1Cropped = []
		tri2Cropped = []
		for i in range(3):
			tri1Cropped.append( (tri1[i][0] - r1[0],  tri1[i][1] - r1[1]) )
			tri2Cropped.append( (tri2[i][0] - r2[0],  tri2[i][1] - r2[1]) )

		# swapping x and y positions
		img1Cropped = pose_image[r1[1]:r1[1]+r1[3], r1[0]: r1[0] + r1[2] ]
		
		# Find the affine transform.
		warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
		
		# warp pixels using affine transformation
		# Apply the Affine Transform just found to the src image
		img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
		
		# Copy triangle to final texture image using a mask 
		
		# Get mask by filling triangle
		mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
		cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);
		
		# Apply mask to cropped region
		img2Cropped = img2Cropped * mask
		
		# Copy triangular region of the rectangular patch to the output image
		tex_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = tex_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
		tex_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = tex_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

	return tex_dst






















































