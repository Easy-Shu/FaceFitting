# Import classes
from Classes.pose import Pose
from Classes.facemodel import FaceModel
from Classes.core import Core

# Other function files
import auxiliary.loader as loader
import auxiliary.draw as draw
import auxiliary.texture as texture
import auxiliary.objHandler as obj
import auxiliary.orthogonalCamera as camera
import glob

import numpy as np
import cv2

#	Output file names
out_folder = 'out' 	# Folder
file_name = 'out' 	# OBJ and MTL file names
texture_name = 'texture.png'

obj_dest = out_folder +'/'+ file_name + '.obj'
tex_dest = out_folder +'/'+ texture_name
mtl_dest = out_folder +'/'+ file_name + '.mtl'

# 		FITTING PARAMETERS DESCRIPTION 
#	target_folder			:	Folder containing images and pts files for a specific target
#
#	image_type				: 	Type of image inside target folder. Only single type accepted.
#
# 	fitting_iterations  	:	Number of fitting iterations
#
#	SD_constraint			: 	Standard Deviations from mean face to constrain fitting solution
#
#	num_components			:	Number of principal components used to do the fitting
#
#	contour_tolerance		:	Contour tolerance, i.e., the size of the radius around each landmark point to look for projected vertices to do the match. 
#
# 	visible_only_matching 	:	Whether or not to compute visible vertices when fitting the contour. Ideally it should be done, but it takes much longer and in some cases it might not make a lot of difference. 
#
#	fixed_landmark_indices	:	Indices of landmarks in the face, excluding occluding landmarks. The correspond to those inside 'target.pts'
#	flipY					: 	Set to true if landmarks in image are set with y-down axis convention

target_folder = 'target4'
image_type = 'jpg'
fitting_iterations = 6
SD_constraint = 2.5
num_components = 70
contour_tolerance = 0.06
visible_only_matching = True
fixed_landmark_indices = list(range(1 , 90+1))
flipY = True

# OBJ files
obj_scr = 'share/mean_face.obj'
base_texture_scr = 'share/texture4K.png'

#	INITIALIZING 
print('Initializing.')

# Files paths
target_points_path = target_folder + '/*.pts'
img_path = target_folder + '/*.' + image_type
BFM_points_path = 'share/BFM.pts'
bfm_path = 'share/01_MorphableModel.mat'
bfm_edgestructure_path = 'share/BFMedgestruct.mat'

#	READ DATA (images, landmarks, face model ...)
print('Reading data.')
images = loader.readImages(img_path)

# Target pts paths
pts_files = glob.glob(target_points_path)

# Load Basel Face Model
bfm = FaceModel(bfm_path, bfm_edgestructure_path, BFM_points_path)

# Initialize Pose static members
Pose.staticInitialization(bfm.meanFace, fixed_landmark_indices, contour_tolerance, visible_only_matching)


# 	INITIALIZE POSES 
print('Initializing poses and executing initial camera estimation.')
poses = []

if len(pts_files) != len(images):
	raise NameError('Missing files.')
	
for i in range(len(pts_files)):
	
	p = Pose(bfm.model_vertices, pts_files[i], images[i])
	
	points = p.getLandmarkPoints()
	
	img = draw.drawKeypoints(p.image, points, size=0.2)
	
	print('Pose %d:\n%s\n' % (i+1, pts_files[i]))
	print('Close image frame to continue.')
	cv2.imshow('imagem', img)
	cv2.waitKey(0)
	
	poses.append(p)

#input('Continue?')
	#	FITTING SHAPE
newMesh = Core.fitIterated_linear(fitting_iterations, bfm, poses, num_components, SD_constraint)

#Showing result of first pose
v, p = poses[0].getPointVertexCorrespondences()
img = draw.drawVertices(poses[0].image, v, newMesh, poses[0].R, poses[0].t, poses[0].s, size= 0.3)
#cv2.imshow('imagem', img)
#cv2.waitKey(0)


	# 	TEXTURE EXTRACTION
	
# Loading obj template for UV coords
vertices, normals, uv_coords, FV, FN, FT, header, materialHeader = obj.readOBJ(obj_scr)

#input('Extract texture?')

# Texture extraction
print('Extracting texture')
base_texture = texture.readTexture(base_texture_scr)
extracted_texture = Core.composeTexture(newMesh, base_texture, poses, uv_coords, FV, FT)

	#	SAVE RESULT
cv2.imwrite(tex_dest, extracted_texture)
obj.writeOBJ(obj_dest, newMesh, normals, uv_coords, FV, FN, FT, header, materialHeader)
obj.writeMTL(out_folder, texture_name)











