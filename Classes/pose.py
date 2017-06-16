###		Pose
# This class holds all the attributes for a single face pose. This includes the target image, 2D landmarks coordinates, camera parameters estimation for given pose, fixed vertex indices as well as estimated occluding vertex indices given estimated camera and current 3D face model. 

# obs: The current mesh will be the same for all pose objects, which means the class should be used to fit to only one specfici target each time the program is executed. 

import auxiliary.loader as loader
import auxiliary.orthogonalCamera as camera
import auxiliary.edgeCorrespondence as edges
import auxiliary.util as util

class Pose():
	
	# 	Class members:
	currentMesh = None
	fixed_landmark_indices = None
	contour_tolerance = 0.020
	visibleOnly = True
	
	# 	Object members:
	# self.image : RGB face image
	# self.imageHeight, self.imageWidth, self.imageCh
	
	# self.R, self.t, self.s : current camera estimation 
	
	# self.occluding_points : list of landmark locations in image that do not have a specific corresponding vertex index in the model (vertex index was set to 0 in the .pts file containing the image landmark positions). 
	
	# self.fixed_points : list of landmark locations in image [(x1,y1), (x2,y2) ... ]
	# self.fixed_vertices_i : list of landmark vertex indices in the model [v1, v2, v3]
	# obs: fixed_vertices_i[i] corresponds to fixed_points[i]
	
	# self.contour_points : almost identical to self.occluding_points, except it discarts points whose model correspondences were not found
	# self.contour_vertices : list of vertex indices that correspond to the contour_points (contour_points[i] <=> contour_vertices[i])
	
	@staticmethod
	def staticInitialization(currentMesh, landmark_indices, tolerance, visible_only):
		Pose.setCurrentMesh(currentMesh)
		Pose.setFixedLandmarkIndices(landmark_indices)
		Pose.setContourTolerance(tolerance)
		Pose.setVisibleOnlyVertices(visible_only)
	
	@staticmethod
	def setCurrentMesh(newMesh):
		Pose.currentMesh = newMesh
		
	@staticmethod
	def setFixedLandmarkIndices(indices):
		Pose.fixed_landmark_indices = indices
	
	# Set the radius of the landmark matcher. Landmarks whose projected vertices onto the image lie outside this radius are disconsidered. 
	# Obs: This is not actually the radius itself because it would vary with the image resolution. The radius is given by:
	# 	Radius = (imageWidth + imageHeight)/2 * tolerance

	@staticmethod
	def setContourTolerance(tolerance):
		Pose.contour_tolerance = tolerance
		
	
	@staticmethod
	def setVisibleOnlyVertices(visibleOnly):
		Pose.visibleOnly = visibleOnly
		
	# Both currentMesh and fixed_landmark_indices should be set before initializing any Pose object.
	def __init__(self, model_vertices, target_points_path, image, flippedY = True):
	
		if Pose.currentMesh is None or Pose.fixed_landmark_indices is None:
			raise NameError('Pose object instantiation without setting static variables is disallowed.')
			
		self.image = image
		self.imageHeight, self.imageWidth, self.imageCh = image.shape
		
		# READ LANDMARKS
		#	Reads image landmarks positions (fixed and occluding)
		image_points_with_index, self.occluding_points = loader.readImageLandmarks(target_points_path)
		
		# LANDMARKS CORRESPONDENCES
		
		#	Find fixed landmarks correspondences:
		self.fixed_vertices_i, self.fixed_points = util.fixedLandmarkCorrespondence(model_vertices, image_points_with_index, Pose.fixed_landmark_indices)
		
		#	Fix y axis flip
		if flippedY:
			self.fixed_points = util.flipY(self.fixed_points, self.imageHeight)
			self.occluding_points = util.flipY(self.occluding_points, self.imageHeight)
		
		# POSE ESTIMATION
		#	Initial pose fit
		self.R, self.t, self.s = camera.estimatePOS(self.fixed_points, self.fixed_vertices_i, Pose.currentMesh)
		
		# CONTOUR LANDMARKS CORRESPONDENCES: for now we initialize contour correspondences to empty lists
		self.contour_vertices = []
		self.contour_points = []
		
		self.min_contour_dist = ((self.imageHeight + self.imageWidth)/2) * Pose.contour_tolerance
		
		self.silhouettes = []
		
	# Computes silhouettes only
	def computeSilhouetteCorrespondence(self, faceModel, recomputeContour = True):
		# Find all silhouettes
		if recomputeContour:
			silhouettes = edges.silhouetteVertices(Pose.currentMesh, faceModel.tl, self.R, Pose.visibleOnly)
			self.silhouettes = silhouettes
		
		# We'll treat them just like contours
		self.contour_vertices, self.contour_points = edges.contourCorrespondences(Pose.currentMesh, self.R, self.t, self.s, self.occluding_points, self.silhouettes, min_dist = self.min_contour_dist)
		
	# Computes the occluding vertices correspondences that were left empty in the constructor
	def computeContourCorrespondences(self, faceModel):
	
		# CONTOUR ESTIMATION
		#	1. Calculate occluding vertices
		occluding_vertices_i = edges.occludingBoundaryVertices(Pose.currentMesh, faceModel.tl, faceModel.Ef, faceModel.Ev, self.R)		
		
		#	2. Find contour matches given the occluding vertices calculated above
		self.contour_vertices, self.contour_points = edges.contourCorrespondences(Pose.currentMesh, self.R, self.t, self.s, self.occluding_points, occluding_vertices_i, min_dist = self.min_contour_dist)
	
	# This method updates the camera. Use this when the currentMesh has been updated to get a more precise camera parameter estimation
	def computeCameraParams(self):
		v, p = self.getPointVertexCorrespondences()
		self.R, self.t, self.s = camera.estimatePOS(p, v, Pose.currentMesh)
	
	# Returns 2 lists. The first one contains the landmark locations in the image and the seconds one has the indices of the vertices in the model that correspond to those points. Some items of these lists are fixed (a.k.a. fixed_vertices_i, fixed_points...) so they won't change during the program. The contour correspondences however will vary based on the camera parameters estimated as well as the face model estimated. 
	def getPointVertexCorrespondences(self):
	
		# FIXED LANDMARKS + CONTOUR
		# duplicate list so we don't change fixed landmarks
		vertex_indices = list(self.fixed_vertices_i)
		vertex_indices.extend(self.contour_vertices)
		
		# same for points
		points = list(self.fixed_points)
		points.extend(self.contour_points)
		
		return vertex_indices, points 
		
	def getLandmarkPoints(self):
		res = list(self.fixed_points)
		res.extend(self.occluding_points)
		
		return res
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		