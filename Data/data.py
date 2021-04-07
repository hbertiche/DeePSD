import os
from random import shuffle
from math import floor
from scipy import sparse
from scipy.spatial.transform import Rotation as R

from time import time
import tensorflow as tf
import tensorflow_graphics as tfg

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from util import *
from IO import *
from values import *

class Data:
	# Reduce I/O operations at the cost of RAM
	_use_buffer = True
	_T_buffer = {}
	_F_buffer = {}
	_E_buffer = {}
	_L_buffer = {}
	
	_f2o = { # fabric to one hot
		'cotton': [0, 0, 0, 1],
		'denim': [0, 0, 1, 0],
		'silk': [0, 1, 0, 0],
		'leather': [1, 0, 0, 0]
	}
									
	def __init__(self, txt, batch_size=10, mode='train'):
		# txt: path to .txt file with list of samples
		# batch_size: batch size
		# mode: 'train' for shuffle
	
		# Read sample list
		self._txt = txt
		samples = self._read_txt()
		# batch size
		self._batch_size = batch_size
		# Shuffle
		if mode == 'train':
			shuffle(samples)
		self._mode = mode
		self._samples = samples
		# Init
		self._idx = 0
	
	def _read_txt(self):
		# samples in txt as: "[SAMPLE_PATH]\t[N.FRAME]\n"
		with open(self._txt, 'r') as f:
			T = f.readlines()
		return [t.replace('\n','').split('\t') for t in T]
	
	# reset when all samples read
	def _reset(self):
		self._idx = 0
		if self._mode == 'train': shuffle(self._samples)
	
	def _read_outfit(self, sample, n, info):
		path = SRC + sample + '/'
		path_preprocess = SRC_PREPROCESS + sample + '/'
		# read outfit garments in order
		with open(path_preprocess + 'outfit_verts.txt','r') as f:
			outfit = [t.replace('\n','').split('\t') for t in f.readlines()]
		garments = [o[0] for o in outfit]
		subindices = [0] + [int(o[1]) for o in outfit]
		# read each garment vertices and concatenate
		V = np.concatenate([readPC2Frame(path + g + '.pc16', n, True) for g in garments], axis=0)
		# template
		if path_preprocess in self._T_buffer: 
			T = self._T_buffer[path_preprocess]
		else:
			T = readPC2(path_preprocess + 'rest.pc16', True)['V'][0]
			if self._use_buffer: self._T_buffer[path_preprocess] = T
		# faces
		if path_preprocess in self._F_buffer: 
			F = self._F_buffer[path_preprocess]
		else:
			F = readFaceBIN(path_preprocess + 'faces.bin')
			if self._use_buffer: self._F_buffer[path_preprocess] = F
		# edges
		if path_preprocess in self._E_buffer: 
			E = self._E_buffer[path_preprocess]
		else: 
			E = readEdgeBIN(path_preprocess + 'edges.bin')
			if self._use_buffer: self._E_buffer[path_preprocess] = E
		# laplacian
		if path_preprocess in self._L_buffer: 
			L = self._L_buffer[path_preprocess]
		else:
			L = sparse.load_npz(path_preprocess + 'laplacian.npz').tocoo()
			if self._use_buffer: self._L_buffer[path_preprocess] = L
		# fabric
		fabric = self._fabric_to_one_hot(info, garments, subindices)
		return V, T, F, E, L, fabric, garments, subindices

	# encode fabric as per-vertex one-hot
	def _fabric_to_one_hot(self, info, garments, subindices):
		_F = None
		for i in range(1, len(subindices)):
			s, e = subindices[i - 1], subindices[i]
			fabric = np.tile(
						np.float32(self._f2o[info['outfit'][garments[i - 1]]['fabric']]).reshape((1, 4)),
						[e - s, 1]
					)
			if _F is None: _F = fabric
			else: _F = np.concatenate((_F, fabric), axis=0)
		return _F
		
	# Get next outfit
	def next_sample(self):
		# read idx and increase
		idx = self._idx
		self._idx += 1
		# reset if all samples read
		if self._idx + 1 > len(self._samples): self._reset()
		# Read sample
		sample, nframe = self._samples[idx]
		nframe = int(nframe)
		# Load info
		info = loadInfo(SRC + sample + '/info.mat')
		# info['poses'] can have shape (72,N) or (72,)
		if len(info['poses'].shape) > 1: P = info['poses'][:,nframe]
		else:
			assert nframe == 0 # sanity check
			P = info['poses']
		# shape&gender
		S = info['shape']
		G = info['gender']
		# weights prior (for unsupervised)
		W = np.load(SRC_PREPROCESS + sample + '/weights.npy')
		# Load data
		V, T, F, E, L, fabric, garments, subindices = self._read_outfit(sample, nframe, info)
		tightness = np.float32(info['tightness'])
		return V, T, L, F, E, P, S, G, W, fabric, tightness, garments, subindices

	# Get a batch of outfits
	def next(self):
		samples = [self.next_sample() for _ in range(self._batch_size)]
		return self._merge_samples(samples)			

	# Merges meshes into single one
	def _merge_samples(self, samples):
		V, indices = self._merge_verts(s[0] for s in samples)
		T, _ = self._merge_verts(s[1] for s in samples)
		L = self._merge_laplacians([s[2] for s in samples], indices)
		F = self._merge_topology([s[3] for s in samples], indices)
		E = self._merge_topology([s[4] for s in samples], indices)
		P = np.stack([s[5] for s in samples])
		S = np.stack([s[6] for s in samples])
		G = np.stack([s[7] for s in samples])
		W, _ = self._merge_verts([s[8] for s in samples])
		fabric, _ = self._merge_verts(s[9] for s in samples)
		tightness = np.stack([s[10] for s in samples])
		outfits = [s[11] for s in samples]
		subindices = [s[12] for s in samples]
		return {
			'vertices': V,
			'template': with_normals(T, L, F),
			'laplacians': L,
			'faces': F,
			'edges': E,
			'poses': P,
			'shapes': S,
			'genders': G,
			'weights_prior': W,
			'fabric': fabric,
			'tightness': tightness,
			'indices': indices,
			'outfits': outfits,
			'subindices': subindices
		}		
	
	def _merge_verts(self, Vs):
		V = None
		indices = [0]
		for v in Vs:
			if V is None: V = v
			else: V = np.concatenate((V,v), axis=0)
			indices += [V.shape[0]]
		return V, indices
	
	def _merge_laplacians(self, Ls, indices):
		idx, data = None, None
		shape = [indices[-1]]*2
		for i,l in enumerate(Ls):
			if idx is None: idx = np.mat([l.row, l.col]).transpose()
			else:
				_idx = np.mat([l.row, l.col]).transpose()
				_idx += indices[i]
				idx = np.concatenate((idx, _idx))
			if data is None: data = l.data
			else: data = np.concatenate((data, l.data))
		return tf.SparseTensor(idx.astype(np.float32), data.astype(np.float32), shape)
	
	def _merge_topology(self, Fs, indices):
		F = None
		for i,f in enumerate(Fs):
			if F is None: F = f
			else: F = np.concatenate((F, f + indices[i]))
		return F