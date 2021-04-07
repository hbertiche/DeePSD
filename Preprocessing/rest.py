import os
import sys
import numpy as np
from math import cos, sin

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from IO import readOBJ, writeFaceBIN, writePC2, loadInfo
from values import *

"""
Converts rest garments (OBJ files) into binary files for faster I/O during training.
Also solves an issues with z-rotation in CLOTH3D rest garments.
"""

def zRotMatrix(zrot):
	c, s = cos(zrot), sin(zrot)
	return np.array([[c, -s, 0],
					 [s,  c, 0],
					 [0,  0, 1]], np.float32)

def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
		else: sys.exit()
	return np.array(out, np.int32)

samples = os.listdir(SRC)
N = len(samples)
for i,sample in enumerate(samples):
	print("Sample " + str(i+1) + '/' + str(N))
	src = SRC + sample + '/'
	dst = SRC_PREPROCESS + sample + '/'
	if os.path.isfile(dst + 'faces.bin'):
		assert os.path.isfile(dst + 'rest.pc16'), 'Faces but no Rest: ' + sample
		continue
	if not os.path.isdir(dst): os.mkdir(dst)
	# list garments
	with open(dst + '/outfit_verts.txt', 'r') as f:
		garments = [t.replace('\n','').split('\t')[0] for t in f.readlines()]
	# read sample
	zrot = loadInfo(src + 'info.mat')['zrot']
	V, F = None, None
	for g in garments:
		v, f = readOBJ(src + g + '.obj')[:2]
		f = quads2tris(f)
		if V is None:
			V = v
			F = f
		else:
			n = V.shape[0]
			V = np.concatenate((V, v), axis=0)
			F = np.concatenate((F, f + n), axis=0)
	V = (zRotMatrix(-zrot) @ V.T).T
	writePC2(dst + 'rest.pc16', V[None], True)
	writeFaceBIN(dst + 'faces.bin', F)