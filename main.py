import fitting.loader as loader
import fitting.orthogonalCamera as camera
import debugging.draw as drawer

import numpy as np
import cv2

BFM_points_path = 'share/BFM.pts'
avg_face_path = 'share/avg_face.ply'
target_points_path = 'data/target.pts'

# Read average face
vertex_list, face_list = loader.readPLY(avg_face_path)

Points, Vertices, indices = loader.readLandmarks(BFM_points_path, vertex_list, target_points_path)

R, t, s = camera.estimatePOS(Points,Vertices)

# Debugging

# compute positions of vertices given the estimated camera
points_from_vertices = []
for v in Vertices:
	p = camera.SOP(v, R, t, s)
	points_from_vertices.append(p)

image_file = 'data/target.png'
img = cv2.imread(image_file)

img_drawn = drawer.drawKeypointsWithIndices(img, points_from_vertices, indices)
cv2.imshow('projected points', img_drawn)
cv2.waitKey(0)

point = camera.SOP(Vertices[0], R, t, s)
