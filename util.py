import os
import sys
import numpy as np
from scipy import sparse
import tensorflow as tf
import tensorflow_graphics as tfg

def rodrigues(r):
	theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
	# avoid zero divide
	theta = np.maximum(theta, np.finfo(np.float64).tiny)
	r_hat = r / theta
	cos = np.cos(theta)
	z_stick = np.zeros(theta.shape[0])
	m = np.dstack([
	  z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
	  r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
	  -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
	).reshape([-1, 3, 3])
	i_cube = np.broadcast_to(
	  np.expand_dims(np.eye(3), axis=0),
	  [theta.shape[0], 3, 3]
	)
	A = np.transpose(r_hat, axes=[0, 2, 1])
	B = r_hat
	dot = np.matmul(A, B)
	R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
	return R
	
def model_summary(targets):
	print("")
	_print = lambda x: print('\t' + x)
	sep = '---------------------------'
	total = 0
	_print(sep)
	_print('MODEL SUMMARY')
	_print(sep)
	for t in targets:
		_print(t.name + '\t' + str(t.shape))
		total += np.prod(t.shape)
	_print(sep)
	_print('Total params: ' + str(total))
	_print(sep)
	print("")

def with_normals(T, L, F):
	T_VN = tfg.geometry.representation.mesh.normals.vertex_normals(T, F)
	return tf.concat((T, T_VN), axis=-1)
	
def sparse_to_tensor(L):
	idx = np.mat([L.row, L.col]).transpose().astype(np.float32)
	data = L.data
	shape = L.shape
	return tf.SparseTensor(idx, data, shape)
	
def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
		else: sys.exit()
	return np.array(out, np.int32)

def faces2edges(F):
	E = set()
	for f in F:
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			E.add(tuple(sorted([f[i], f[j]])))
	return np.int32(list(E))

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
		data += [1 / n] * n
	return sparse.coo_matrix((data, (row, col)), shape=[len(G)] * 2)