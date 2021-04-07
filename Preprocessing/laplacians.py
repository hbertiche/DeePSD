import numpy as np
from scipy import sparse

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from IO import *
from values import *

"""
Precomputes each outfit laplacian matrix for faster training time.
"""
	
def faces2edges(F):
	E = set()
	for f in F:
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			E.add(tuple(sorted([f[i], f[j]])))
	return E

def edges2graph(E):
	G = {}
	for e in E:
		if not e[0] in G: G[e[0]] = {}
		if not e[1] in G: G[e[1]] = {}
		G[e[0]][e[1]] = 1
		G[e[1]][e[0]] = 1
	return G

def laplacianMatrix(F):
	E = faces2edges(F)
	G = edges2graph(E)
	row, col, data = [], [], []
	for v in G:
		n = len(G[v])
		row += [v] * n
		col += [u for u in G[v]]
		data += [1.0 / n] * n
	return sparse.coo_matrix((data, (row, col)), shape=[len(G)] * 2)
	
samples = os.listdir(SRC_PREPROCESS)
N = len(samples)
for i,sample in enumerate(samples):
	print(str(i+1) + " / " + str(N))
	src = SRC_PREPROCESS + sample + '/'
	if os.path.isfile(src + 'laplacian.npz'): continue
	fpath = src + 'faces.bin'
	F = readFaceBIN(fpath)
	L = laplacianMatrix(F)
	sparse.save_npz(src + 'laplacian.npz', L)