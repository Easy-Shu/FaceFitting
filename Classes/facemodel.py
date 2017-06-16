### 	FaceModel
# An instance of this class holds the main attributes for the face model, which includes mean face, principal components, standard deviations, edge vertices and edge faces list, etc. 

import auxiliary.loader as loader

import numpy as np

class FaceModel():
	
	# 	Object Members:
	
	# self.shapePC
	# self.shapeEV
	# self.meanFace 
	# self.tl
	# self.Ev
	# self.Ef
	# self.model_vertices
	
	def __init__(self, bfm_path, bfm_edgestructure_path, BFM_points_path):
		
		# BFM file		
		# Load Basel Face Model Mat 
		mat = loader.loadMat(bfm_path)
		
		#	Principal Components, (3n x 199)
		self.shapePC = mat['shapePC']

		#	Standard Deviations for each component
		self.shapeEV = mat['shapeEV']

		#	Mean face, same as average face, except the format is ndarray instad of list of tuples, which is easier to use for this
		self.meanFace = mat['shapeMU']

		#	Triangles
		self.tl = mat['tl']
		
		# We need to swap col0 and col1 because they inverted them and so normal calculations won't work
		temp = 	np.copy(self.tl[:, 0])
		self.tl[:, 0] = self.tl[:, 1]
		self.tl[:, 1] = temp
		
		# Edges file
		#	Load edge structures
		edge_mat = loader.loadMat(bfm_edgestructure_path)

		#	num(edges) by 2 matrix storing vertices adjacent to each edge
		self.Ev = edge_mat['Ev']

		#	num(edges) by 2 matrix storing faces adjacent to each edge
		self.Ef = edge_mat['Ef']
		
		# Reads model landmarks vertex indices
		# Returned value holds both model vertex indices and landmark vertex indices 
		self.model_vertices = loader.readModelLandmarks(BFM_points_path)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		