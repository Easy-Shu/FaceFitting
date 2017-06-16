import auxiliary.util as util
import numpy as np

# Estimate scaled orthographic projection parameters from 2D-3D correspondences
# Based on 3DMM Edges Master POS function
def estimatePOS(points2D, vertex_indices, currentMesh):
	
	# Both lists have to be the same size
	if len(points2D) != len(vertex_indices):
		raise NameError( r"Correspondences don't match")
		
	# Convert from vertex indices to 3D vertices
	vertices3D = util.verticesFromCurrentMesh(vertex_indices, currentMesh)
	
	num_correspondences = len(points2D)
	
	A = np.zeros((2*num_correspondences, 8))
	d = np.zeros((2*num_correspondences, 1))
	
	for i in range(0, 2*num_correspondences, 2):
		p = list(points2D[i//2])
		v = list(vertices3D[i//2])
		A[i, 0:3] = v
		A[i, 3] = 1
		
		A[i+1, 4:7] = v
		A[i+1, 7] = 1
		
		d[i:i+2,0] = p
		
	# Solve linear system using least square approach
	x = np.linalg.lstsq(A, d)[0]
	
	R1 = x[0:3,0]
	R2 = x[4:7,0]
	
	# Translation vector
	sTx = x[3,0]
	sTy = x[7,0]
	
	# Scale
	normR1 = np.linalg.norm(R1)
	normR2 = np.linalg.norm(R2)
	s = (normR1+normR2)/2
	
	# Normalize R1 and R2, find r3
	r1 = R1/normR1
	r2 = R2/normR2
	r3 = np.cross(r1,r2)
	
	USV = np.vstack([r1,r2,r3])
	
	# Calculate SVD
	# obs: the V returned here is already transposed (V = v.t)
	U, S, V = np.linalg.svd(USV)
	
	R = np.matmul(U, V)
	
	if np.linalg.det(R) < 0:
		U[2,:] *= -1
		R = np.matmul(U, V)
		
	# Remove scale
	Tx = sTx/s
	Ty = sTy/s
	t = np.array([Tx, Ty])
	
	return R, t, s

	
def SOP(vertex, R, t, s):
	orth = np.array([[1,0,0],[0,1,0]])
	orth = orth * s
	
	sT = s * t
	
	Result = np.matmul( np.matmul(orth, R), vertex) + sT
	
	return Result
	
	
	
	
	
	
	
	