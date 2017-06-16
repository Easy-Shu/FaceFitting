import auxiliary.loader as loader
import auxiliary.orthogonalCamera as camera
import auxiliary.util as util

import numpy as np
from scipy.optimize import lsq_linear
from scipy.optimize import minimize

# Estimates PCA weights to fit shape given the camera position and parameters usign Scaled Orthographic Projection.
# It uses a constrained linear system solver to make sure the solution is plausible. More specifically, the solution x has to be in k standard deviations from the mean.
# 'Single' means it only does the fit to a single head position, as opposed to finding the best set of PCA weights given a set of multiple positions
#	[in] meanFace: mean face with vertices in 2D array format size [3n x 1], n = number of vertices
#	[in] shapePC: principal components of cov. matrix size [3n x m], m = number of faces in the dataset, which is also the max number of PCA coefficients. m = 199 for BFM
#	[in] shapeEV: standard deviations (or variance? have to check BFM) of each component size [m x 1], which is the same as the eigenvalues of the covariance matrix along the principal components
#	[in] vertex_indices: index of vertices at the mean shape that will be fit, corresponding to the landmarks in xp
#	[in] xp: list of tuples of 2D points of the landmark positions in the image
#	[in] Rotation: Camera rot
#	[in] Scale: Camera focal length divided by mean distance 
#	[in] Translation: [Tx, Ty]
#	[in] num_components: number of PCA coefficients used to fit the model. 1 <= num_components <= m
#	[in] numsd: number of standard deviations from mean for solution

def fitSingleFaceLinear(meanFace, shapePC, shapeEV, vertex_indices, points, Rotation, Translation, Scale, num_components, numsd):
	
	# L: Number of landmarks
	L = len(points)
	
	# Build affine camera matrix
	t = Scale * Translation
	R = Scale * Rotation
	
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
	reducedShapePC = shapePC[:, 0: num_components]
	
	# Build BasisCorr matrix with the corresponding principal components
	BasisCorr = np.zeros((4*L, num_components))
	for i in range(L):
		basis_row = reducedShapePC[vertex_indices[i]*3 :  vertex_indices[i]*3+3, :]
		BasisCorr[4*i : 4*i + 3, :] = basis_row
		
	# Build MeanCorr with the corresponding 3D vertices from the mean face in homogeneous form
	MeanCorr = np.ones((4*L, 1))
	for i in range(L):
		vertex = meanFace[vertex_indices[i]*3 : vertex_indices[i]*3+3, 0]
		MeanCorr[4*i : 4*i + 3, 0] = vertex
		
	# Linear system min Ax - b 
	A = np.matmul(P, BasisCorr)
	b = -(np.matmul(P, MeanCorr) - homogeneous2DPoints)
	
	# Constraints 
	bounds = numsd * shapeEV[0:num_components, 0]
	lb = -bounds
	up = bounds
	
	# Solve system
	res = lsq_linear(A, b.flatten(), bounds=(lb, up))
	
	x = res['x']
	cost = res['cost']
		
	return x, cost

# Same as above, only it uses non linear solving method. Basically it means we're not solving a simple system of linear equations just as above, but also we're directly penalizing solutions whose norm is too big, meaning it is far from the "mean" face, instead of just imposing a constraint to the solutions. This is because more often than not the "best" solution just falls into the imposed bounds, meaning the algorithm tries to generate a non-face-like mesh in order to minimize the error. 
# The drawback is that it might take longer and solution might not be optimal 
# obs: this function make use of nonLinearCost as its cost function
def fitSingleFace_NonLinear(meanFace, shapePC, shapeEV, vertex_indices, xp, Rotation, Translation, Scale, num_components, numsd, beta):
	
	# L: Number of landmarks
	L = len(xp)
	
	# Build affine camera matrix
	t = Scale * Translation
	R = Scale * Rotation
	
	AffineCameraMat = np.zeros((3,4))
	AffineCameraMat[0:2, 0:3] = R[0:2, 0:3]
	AffineCameraMat[0:2, 3] = t
	AffineCameraMat[2,3] = 1
	
	# Build system camera matrix (stack camera matrices in the diagonal)
	P = np.zeros((3*L, 4*L))
	for i in range(L):
		P[3*i: 3*i + 3, 4*i : 4*i + 4] = AffineCameraMat
		
	# Build 2D points vector
	homogeneous2DPoints = util.pointsTo2DArray_homogeneous(xp)
	
	# Reduce principal components to the number of components set to be used to fit the model
	reducedShapePC = shapePC[:, 0: num_components]
	
	# Build BasisCorr matrix with the corresponding principal components
	BasisCorr = np.zeros((4*L, num_components))
	for i in range(L):
		basis_row = reducedShapePC[vertex_indices[i]*3 :  vertex_indices[i]*3+3, :]
		BasisCorr[4*i : 4*i + 3, :] = basis_row
		
	# Build MeanCorr with the corresponding 3D vertices from the mean face in homogeneous form
	MeanCorr = np.ones((4*L, 1))
	for i in range(L):
		vertex = meanFace[vertex_indices[i]*3 : vertex_indices[i]*3+3, 0]
		MeanCorr[4*i : 4*i + 3, 0] = vertex
		
	# Reduced shapeEV
	reduced_shapeEV = shapeEV[0:num_components, 0]
	
	# We divide by mean SD to "normalize" it since values are too big
	reduced_shapeEV = np.divide(reduced_shapeEV, np.mean(shapeEV))
	
	# Minimize using non-linear cost func.
	initialGuess = np.random.rand(num_components)*1000
	add_args = (P, BasisCorr, MeanCorr, homogeneous2DPoints, reduced_shapeEV, beta)
	
	res = minimize(nonLinearCost, initialGuess, args=add_args)
	
	return res.x
	
def nonLinearCost(x, P, BasisCorr, MeanCorr, homogeneous2DPoints, shapeEV, beta):
	# Compute shape cost
	A = np.matmul(P, BasisCorr)
	avg = np.matmul(P, MeanCorr)
	Face = np.matmul(A, x.flatten()) + avg
	
	FaceDiff = Face - homogeneous2DPoints
	shapeCost = np.linalg.norm(FaceDiff)
	
	# Compute solution cost
	meanPenalizer = np.linalg.norm(np.divide(x.flatten(), shapeEV) )
	
	return shapeCost + beta*meanPenalizer
	

# Returns the shape vertices given the PCA parameters x
def shapeFromParams(meanFace, shapePC, x):
	num_comp = x.shape[0]
	
	new_shapePC = shapePC[:,0:num_comp]
	meshVertices = np.matmul(new_shapePC, x)
	
	# if we don't reshape it sums wrongly and gives memory error
	meshVertices.shape = (-1,1)
	
	return meshVertices + meanFace
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
	