### 	Core 
# Contains the main methods for fitting etc. 
# It is basically a class with static methods only. For organization purposes I used a class here, but it is definately not necessary.
from Classes.pose import Pose

import auxiliary.loader as loader
import auxiliary.orthogonalCamera as camera
import auxiliary.util as util
import auxiliary.draw as draw
import auxiliary.texture as texture

import numpy as np
import cv2
from scipy.optimize import lsq_linear

class Core():
	computeNewContour_iterations = 3
	
	# Fit the model shape using fitPoses_linear iteratively
	# Assumes Pose has already been initialized
	@staticmethod
	def fitIterated_linear(iterations, faceModel, poses, num_components, numsd):
		
		print('Starting fitting.')
			
		print('Computing new camera position')
		for p in poses:
			p.computeCameraParams()
		
		# Initial shape estimation
		print('Initial shape estimation...')
		x, cost = Core.fitPoses_linear(faceModel, poses, num_components, numsd)
		newMesh = Core.shapeFromParams(x, faceModel)
		Pose.setCurrentMesh(newMesh)
		
		# Iterated fitting
		for i in range(iterations):
			print('\nIteration %d/%d\n' % (i+1, iterations) )
			
			print('Computing contour correspondences')
			recomputeContour = False
			if (i) % Core.computeNewContour_iterations == 0:
				print('Recomputing visible contour')
				recomputeContour = True
				
			for p in poses:
				p.computeSilhouetteCorrespondence(faceModel, recomputeContour=recomputeContour)
					
			print('Computing new camera position')
			for p in poses:
				p.computeCameraParams()
				
			print('Solving system')
			x, cost = Core.fitPoses_linear(faceModel, poses, num_components, numsd)
			newMesh = Core.shapeFromParams(x, faceModel)
			
			print('Saving new face mesh\n')
			Pose.setCurrentMesh(newMesh)
			
		return Pose.currentMesh
	
	# Fit the facemodel to a set of Pose objects in a single iteration solving a linear system and returning the vector solution. More iterations could be used calling this method more times. 
	
	# 	[in] faceModel : FaceModel object
	# 	[in] poses : List of Pose objects, each representing a different pose of the same target
	#	[in] num_components : number of PCA coefficients used to fit the model. 1 <= num_components <= m
	# 	[in] numsd : number of standard deviations from mean for solution
	
	# obs: Pose objects have some attributes that should be updated each iteration, namely calling computeContourCorrespondences and computeCameraParams. This method assumes those have already been called. 
	
	@staticmethod
	def fitPoses_linear(faceModel, poses, num_components, numsd):
		
		As = []
		bs = []
		
		# Build linear system matrices for each pose, and then arrange them together forming one system
		for pose in poses:
			vertex_indices, points = pose.getPointVertexCorrespondences()
			
			# L: Number of landmarks
			L = len(points)
			
			# Build affine camera matrix
			t = pose.s * pose.t
			R = pose.s * pose.R
			
			AffineCameraMat = np.zeros((3,4))
			AffineCameraMat[0:2, 0:3] = R[0:2, 0:3] 
			AffineCameraMat[0:2, 3] = t 
			AffineCameraMat[2,3] = 1 
			
			# Build system camera matrix (stack camera matrices in the diagonal)
			P = np.zeros((3*L, 4*L))
			for i in range(L):
				P[3*i: 3*i + 3, 4*i : 4*i + 4] = AffineCameraMat
				
			# Build 2D points vector
			homogeneous2DPoints = util.pointsTo2DArray_homogeneous(points) 
			
			# Reduce principal components to the number of components set to be used to fit the model
			reducedShapePC = faceModel.shapePC[:, 0: num_components]
			
			# Build BasisCorr matrix with the corresponding principal components
			BasisCorr = np.zeros((4*L, num_components))
			for i in range(L):
				basis_row = reducedShapePC[vertex_indices[i]*3 :  vertex_indices[i]*3+3, :]
				BasisCorr[4*i : 4*i + 3, :] = basis_row 
				
			# Build MeanCorr with the corresponding 3D vertices from the mean face in homogeneous form
			MeanCorr = np.ones((4*L, 1))
			for i in range(L):
				vertex = faceModel.meanFace[vertex_indices[i]*3 : vertex_indices[i]*3+3, 0]
				MeanCorr[4*i : 4*i + 3, 0] = vertex 
			
			# Linear system min(Ax - b)
			A = np.matmul(P, BasisCorr)
			b = homogeneous2DPoints - np.matmul(P, MeanCorr)
			
			# Save sub system
			As.append(A)
			bs.append(b)
			
		# Concatenate all sub systems into one
		finalA = As[0]
		finalb = bs[0]
		
		for i in range(1, len(As)):
			finalA = np.concatenate((finalA, As[i]))
			finalb = np.concatenate((finalb, bs[i]))
			
		# Constraints 
		bounds = numsd * faceModel.shapeEV[0:num_components, 0]
		lb = -bounds
		up = bounds
		
		# Solve system
		res = lsq_linear(finalA, finalb.flatten(), bounds=(lb, up))
		
		x = res['x']
		cost = res['cost']
		
		return x, cost
	
	# Extracts all the textures from the shapes and compose a single texture	
	# Assumes the texture is square and colored
	@staticmethod
	def composeTexture(estimated_shape, base_texture, poses, uv_coords, FV, FT):
		brightnessRemoval = 0.7
		textures = []
		masks = []
		
		texture_size = base_texture.shape[0]
		
		for p in poses :
			# todo: rough color correction along poses. Regions around landmarks should have similar overall color
		
			extracted_texture, texture_mask = texture.extractTextureSinglePose( estimated_shape, base_texture, p, uv_coords, FV, FT) 
				
			
			textures.append(extracted_texture)
			masks.append(texture_mask)
			
		# Normalize masks. mask_i = mask_i / sum_masks
		# adding small amount avoids divisons by zero
		sum_masks = sum(masks) + 0.001
		for i in range(len(masks)):
			masks[i] = masks[i]/ sum_masks
			
		# Calculate mean texture taking into consideretion the normalized masks.
		# mean_text =  mask_1 * text_1 +...+ mask_i * text_i 
		mean_texture = np.zeros(textures[0].shape)
		
		for i in range(len(textures)):
			mask = masks[i]
			
			# Repeat mask to give 3 channels
			# mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2) # less complicated method below
			mask = np.dstack([mask,mask,mask])
			mean_texture += mask * textures[i]
					 
		
		# We want to reduce shadows and specular highlights. To do this we'll iterate through every texture pixel and compute its brightness and compare to the brightness of the average texture at that pixel. The difference D = |(current texture brightness) - (average texture brightness)| will be in the range [0, 254]. 
		# D/254 will normalize it to [0,1], being 1 total difference. Given that the mask value of that texture at that pixel is W, we'll have: 
		# new W = (W -W*D*brightnessRemoval) = W*(1 - D*brightnessRemoval)
		# brightnessRemoval might be greater than 1, so the new W might become negative. In this case we set its value to 0
		# We'll skip the pixel if W is already 0. 
		
		for y in range(texture_size):
			for x in range(texture_size):
				R, G, B = mean_texture[y, x, :]
				avg_brightness = 0.5 * max(R, G, B) + 0.5*min(R, G, B)
				
				for i in range(len(textures)):
					W = masks[i][y,x]
					if W == 0:
						continue
					
					R, G, B = textures[i][y, x, :]
					tex_brightness = 0.5 * max(R, G, B) + 0.5*min(R, G, B)
					D = abs(avg_brightness- tex_brightness)/254
					newW = W*(1-D*brightnessRemoval)
					masks[i][y,x] = max(0, newW)
					
		# Increase filter contrast by using power of 2. This avoids too much texture overlapping. Maybe even use power of 3.
		for i in range(len(masks)):
			masks[i] = masks[i] * masks[i]
		
		# Normalize masks again
		sum_masks = sum(masks) + 0.001
		for i in range(len(masks)):
			masks[i] = masks[i]/ sum_masks
			
		# Compose final texture:
		final_texture = np.zeros(textures[0].shape)
		for i in range(len(textures)):
			mask = masks[i]
			text = textures[i]
			weighted_text = np.dstack([mask,mask,mask]) * text
			final_texture += weighted_text
			
		return final_texture
		
	
	# Returns the shape vertices given the PCA parameters x
	@staticmethod
	def shapeFromParams(x, faceModel):
		num_comp = x.shape[0]
		
		new_shapePC = faceModel.shapePC[:,0:num_comp]
		meshVertices = np.matmul(new_shapePC, x)
		
		# if we don't reshape it sums wrongly and gives memory error
		meshVertices.shape = (-1,1)
			
		return meshVertices + faceModel.meanFace
			
	def saveMesh(file_path, mesh, faceModel, annotate_vertices=[]):
		draw.writeAnnotatedPLY(file_path, mesh, faceModel.tl, annotate_vertices)
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
