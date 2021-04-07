import os
import numpy as np
from struct import pack, unpack
import scipy.io as sio

def loadInfo(filename):
	'''
	this function should be called instead of direct sio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	'''
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	del data['__globals__']
	del data['__header__']
	del data['__version__']
	return _check_keys(data)

def _check_keys(dict):
	'''
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	'''
	for key in dict:
		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
			dict[key] = _todict(dict[key])
	return dict        

def _todict(matobj):
	'''
	A recursive function which constructs from matobjects nested dictionaries
	'''
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, sio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		elif isinstance(elem, np.ndarray) and np.any([isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
			dict[strg] = [None] * len(elem)
			for i,item in enumerate(elem):
				if isinstance(item, sio.matlab.mio5_params.mat_struct):
					dict[strg][i] = _todict(item)
				else:
					dict[strg][i] = item
		else:
			dict[strg] = elem
	return dict

"""
Reads OBJ files
Only handles vertices, faces and UV maps
Input:
- file: path to .obj file
Outputs:
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data in .obj file, it shall return Vt=None and Ft=None
"""
def readOBJ(file):
	V, Vt, F, Ft = [], [], [], []
	with open(file, 'r') as f:
		T = f.readlines()
	for t in T:
		# 3D vertex
		if t.startswith('v '):
			v = [float(n) for n in t.replace('v ','').split(' ')]
			V += [v]
		# UV vertex
		elif t.startswith('vt '):
			v = [float(n) for n in t.replace('vt ','').split(' ')]
			Vt += [v]
		# Face
		elif t.startswith('f '):
			idx = [n.split('/') for n in t.replace('f ','').split(' ')]
			f = [int(n[0]) - 1 for n in idx]
			F += [f]
			# UV face
			if '/' in t:
				f = [int(n[1]) - 1 for n in idx]
				Ft += [f]
	V = np.array(V, np.float32)
	Vt = np.array(Vt, np.float32)
	if Ft: assert len(F) == len(Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
	else: Vt, Ft = None, None
	return V, F, Vt, Ft

"""
Writes OBJ files
Only handles vertices, faces and UV maps
Inputs:
- file: path to .obj file (overwrites if exists)
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data as input, it will write only 3D data in .obj file
"""
def writeOBJ(file, V, F, Vt=None, Ft=None):
	if not Vt is None:
		assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'
		
	with open(file, 'w') as file:
		# Vertices
		for v in V:
			line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'
			file.write(line)
		# UV verts
		if not Vt is None:
			for v in Vt:
				line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'
				file.write(line)
		# 3D Faces / UV faces
		if Ft:
			F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]
		else:
			F = [[str(i + 1) for i in f] for f in F]		
		for f in F:
			line = 'f ' + ' '.join(f) + '\n'
			file.write(line)

"""
Reads PC2 files, and proposed format PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- float16: False for PC2 files, True for PC16
Output:
- data: dictionary with .pc2/.pc16 file data
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""
def readPC2(file, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	data = {}
	bytes = 2 if float16 else 4
	dtype = np.float16 if float16 else np.float32
	with open(file, 'rb') as f:
		# Header
		data['sign'] = f.read(12)
		# data['version'] = int.from_bytes(f.read(4), 'little')
		data['version'] = unpack('<i', f.read(4))[0]
		# Num points
		# data['nPoints'] = int.from_bytes(f.read(4), 'little')
		data['nPoints'] = unpack('<i', f.read(4))[0]
		# Start frame
		data['startFrame'] = unpack('f', f.read(4))
		# Sample rate
		data['sampleRate'] = unpack('f', f.read(4))
		# Number of samples
		# data['nSamples'] = int.from_bytes(f.read(4), 'little')
		data['nSamples'] = unpack('<i', f.read(4))[0]
		# Animation data
		size = data['nPoints']*data['nSamples']*3*bytes
		data['V'] = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
		data['V'] = data['V'].reshape(data['nSamples'], data['nPoints'], 3)
		
	return data
	
"""
Reads an specific frame of PC2/PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- frame: number of the frame to read
- float16: False for PC2 files, True for PC16
Output:
- T: mesh vertex data at specified frame
"""
def readPC2Frame(file, frame, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	assert frame >= 0 and isinstance(frame,int), 'Frame must be a positive integer'
	bytes = 2 if float16 else 4
	dtype = np.float16 if float16 else np.float32
	with open(file,'rb') as f:
		# Num points
		f.seek(16)
		# nPoints = int.from_bytes(f.read(4), 'little')
		nPoints = unpack('<i', f.read(4))[0]
		# Number of samples
		f.seek(28)
		# nSamples = int.from_bytes(f.read(4), 'little')
		nSamples = unpack('<i', f.read(4))[0]
		if frame > nSamples:
			print("Frame index outside size")
			print("\tN. frame: " + str(frame))
			print("\tN. samples: " + str(nSamples))
			return
		# Read frame
		size = nPoints * 3 * bytes
		f.seek(size * frame, 1) # offset from current '1'
		T = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
	return T.reshape(nPoints, 3)

"""
Writes PC2 and PC16 files
Inputs:
- file: path to file (overwrites if exists)
- V: 3D animation data as a three dimensional array (N. Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""
def writePC2(file, V, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	if float16: V = V.astype(np.float16)
	else: V = V.astype(np.float32)
	with open(file, 'wb') as f:
		# Create the header
		headerFormat='<12siiffi'
		headerStr = pack(headerFormat, b'POINTCACHE2\0',
						1, V.shape[1], 0, 1, V.shape[0])
		f.write(headerStr)
		# Write vertices
		f.write(V.tobytes())

"""
Appends frames to PC2 and PC16 files
Inputs:
- file: path to file
- V: 3D animation data as a three dimensional array (N. New Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""		
def writePC2Frames(file, V, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	# Read file metadata (dimensions)
	if os.path.isfile(file):
		if float16: V = V.astype(np.float16)
		else: V = V.astype(np.float32)
		with open(file, 'rb+') as f:
			# Num points
			f.seek(16)
			nPoints = unpack('<i', f.read(4))[0]
			assert len(V.shape) == 3 and V.shape[1] == nPoints, 'Inconsistent dimensions: ' + str(V.shape) + ' and should be (-1,' + str(nPoints) + ',3)'
			# Read n. of samples
			f.seek(28)
			nSamples = unpack('<i', f.read(4))[0]
			# Update n. of samples
			nSamples += V.shape[0]
			f.seek(28)
			f.write(pack('i', nSamples))
			# Append new frame/s
			f.seek(0, 2)
			f.write(V.tobytes())
	else: writePC2(file, V, float16)
	
"""
Reads proposed compressed file format for mesh topology.
Inputs:
- fname: name of the file to read
Outputs:
- F: faces of the mesh, as triangles
"""
def readFaceBIN(fname):
	if '.' in os.path.basename(fname) and not fname.endswith('.bin'): 
		print("File name extension should be '.bin'")
		return
	elif not '.' in os.path.basename(fname): fname += '.bin'
	with open(fname, 'rb') as f:
		F = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
		return F.reshape((-1,3))
			
"""
Compress mesh topology into uint16 (Note that this imposes a maximum of 65,536 vertices).
Writes this data into the specified file.
Inputs:
- fname: name of the file to be created (provide NO extension)
- F: faces. MUST be an Nx3 array
"""
def writeFaceBIN(fname, F):
	assert type(F) is np.ndarray, "Make sure faces is an Nx3 NumPy array"
	assert len(F.shape) == 2 and F.shape[1] == 3, "Faces have the wrong shape (should be Nx3)"
	if '.' in os.path.basename(fname) and not fname.endswith('.bin'): 
		print("File name extension should be '.bin'")
		return
	elif not '.' in os.path.basename(fname): fname += '.bin'
	F = F.astype(np.uint16)
	with open(fname, 'wb') as f:
		f.write(F.tobytes())

""" EDGES """		
def readEdgeBIN(fname):
	if '.' in os.path.basename(fname) and not fname.endswith('.bin'): 
		print("File name extension should be '.bin'")
		return
	elif not '.' in os.path.basename(fname): fname += '.bin'
	with open(fname, 'rb') as f:
		E = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
		return E.reshape((-1,2))
		
def writeEdgeBIN(fname, E):
	assert type(E) is np.ndarray, "Make sure edges is an Nx2 NumPy array"
	assert len(E.shape) == 2 and E.shape[1] == 2, "Edges have the wrong shape (should be Nx2)"
	if '.' in os.path.basename(fname) and not fname.endswith('.bin'): 
		print("File name extension should be '.bin'")
		return
	elif not '.' in os.path.basename(fname): fname += '.bin'
	E = E.astype(np.uint16)
	with open(fname, 'wb') as f:
		f.write(E.tobytes())	