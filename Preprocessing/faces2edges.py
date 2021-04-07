import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from IO import *
from values import *

"""
Precomputes each sample edges as Nx2 np.arrays and stores them for faster training time.
"""

def faces2edges(F):
	E = set()
	for f in F:
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			E.add(tuple(sorted([f[i], f[j]])))
	return E
	
samples = os.listdir(SRC_PREPROCESS)
N = len(samples)

for i,sample in enumerate(samples):
	print("Sample " + str(i+1) + '/' + str(N))
	if os.path.isfile(SRC_PREPROCESS + sample + '/edges.bin'): continue
	file = SRC_PREPROCESS + sample + '/faces.bin'
	F = readFaceBIN(file)
	E = faces2edges(F)
	E = np.array(list(E))
	file = file.replace('faces','edges')
	writeEdgeBIN(file, E)