import numpy as np

# Read OBJ file and returns data, including vertex positions, vertex normals, texture positions and triangle faces.
# obs: OBJs use index starting 1 convention. We will not subtract it here. 
def readOBJ(file_name):
	print('\nLoading ' + file_name)

	header = None
	materialHeader = ''
	vertices = []
	normals = []
	textures = []
	
	FV = []
	FN = []
	FT = []
	
	vertexToken = 'v'
	normalToken = 'vn'
	textureToken = 'vt'
	faceToken = 'f'
	
	file = open(file_name)
	
	
	while True:
			
		words = file.readline().split()
		
		if len(words) == 0:
			if len(FV) != 0:
				break
			else:
				continue
				
		elif words[0] == '#':
			continue
		
		elif header == None:
			header = ' '.join(words) + '\n'
			
		elif words[0] == vertexToken:
			x = float(words[1])
			y = float(words[2])
			z = float(words[3])

			vertices.append((x, y, z)) 
			
		elif words[0] == normalToken:
			x = float(words[1])
			y = float(words[2])
			z = float(words[3])

			normals.append((x, y, z)) 
			
		elif words[0] == textureToken:
			u = float(words[1])
			v = float(words[2])

			textures.append((u, v))
		
		# Assumes it has vertex, face normals and texture. Otherwise will give error, should check that later.
		elif words[0] == faceToken:
			v1, t1, n1 = [float(x) for x in words[1].split('/')]
			v2, t2, n2 = [float(x) for x in words[2].split('/')]
			v3, t3, n3 = [float(x) for x in words[3].split('/')]
			
			FV.append((v1, v2, v3))
			FN.append((n1, n2, n3))
			FT.append((t1, t2, t3))
		
		elif len(FV) == 0 and len(vertices) != 0:
			materialHeader += ' '.join(words) + '\n'
			
		else: 
			print('Hit unsupported condition')
			print('Line read: %s' % ' '.join(words) + '\n')
			break
		
	file.close()
	
	vertices = np.array(vertices)
	normals = np.array(normals)
	textures = np.array(textures)
	FV = np.array(FV).astype(int)
	FN = np.array(FN).astype(int)
	FT = np.array(FT).astype(int)
	
	return vertices, normals, textures, FV, FN, FT, header, materialHeader
	
# Assumes data is in array format.
def writeOBJ(file_name, vertices, normals, textures, FV, FN, FT, header, materialHeader):	
	print('Saving ' + file_name)
	
	vertexToken = 'v'
	normalToken = 'vn'
	textureToken = 'vt'
	faceToken = 'f'

	V = vertices.reshape((-1,3))
	N = normals.reshape((-1,3))
	T = textures.reshape((-1,2))
	
	num_vertices = V.shape[0]
	num_normals = N.shape[0]
	num_textures = T.shape[0]
	num_faces = FV.shape[0]
	
	file = open(file_name, 'w')
	
	file.write(header)
	
	for i in range(num_vertices):
		x, y, z = V[i, :]
		file.write(vertexToken + ' ' + str(x) + ' ' + str(y) + ' ' + str(z)+'\n')
	
	for i in range(num_normals):
		x, y, z = N[i, :]
		file.write(normalToken + ' ' + str(x) + ' ' + str(y) + ' ' + str(z)+'\n')
	
	for i in range(num_textures):
		u, v = T[i, :]
		file.write(textureToken + ' ' + str(u) + ' '+ str(v)+'\n')
		
	file.write(materialHeader)

	for i in range(num_faces):
		v1, v2, v3 = FV[i, :]
		n1, n2, n3 = FN[i, :]
		t1, t2, t3 = FT[i, :]
		
		f1 = str(v1) + '/' + str(t1) + '/' + str(n1)
		f2 = str(v2) + '/' + str(t2) + '/' + str(n2)
		f3 = str(v3) + '/' + str(t3) + '/' + str(t3)
		
		file.write(faceToken + ' ' + f1 + ' ' + f2 + ' ' + f3 + '\n')
		
		
	file.close()

# Writes MTL file with texture specified name
# Cannot change MTL name because it is 
def writeMTL(out_folder, texture_name):
	mtl_dest = out_folder + '/mean_face.mtl'
	file = open(mtl_dest, 'w')
	file.write('newmtl Default\n')
	file.write('Kd 0.48 0.48 0.48\n')
	file.write('Ns 256\n')
	file.write('d 1\n')
	file.write('illum 2\n')
	file.write('Ka 0 0 0\n')
	file.write('Ks 0.04 0.04 0.04\n')
	file.write('map_Kd '+ texture_name+ '\n')
	file.close()




































































	
	
	
	
	
	
	
	
	
	
	