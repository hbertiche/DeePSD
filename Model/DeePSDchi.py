import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smpl.smpl_np import SMPLModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from Layers import *
from values import rest_pose

class DeePSD:
	# Builds models and initializes SMPL
	def __init__(self, psd_dim, checkpoint=None, rest_pose=rest_pose):
		# psd_dim: dimensionality of blend shape matrices (PSD)
		# checkpoint: path to pre-trained model
		# with_body: will compute SMPL body vertices
		# rest_pose: SMPL rest pose for the dataset (star pose in CLOTH3D; A-pose in TailorNet; ...)
		self._psd_dim = psd_dim
		self._build()
		self._best = float('inf') # best metric obtained with this model
		
		smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/smpl/'
		self.SMPL = {
			0: SMPLModel(smpl_path + 'model_f.pkl', rest_pose),
			1: SMPLModel(smpl_path + 'model_m.pkl', rest_pose)
		}
		# load pre-trained
		if checkpoint is not None:
			print("Loading pre-trained model: " + checkpoint)
			self.load(checkpoint)

	# Builds model
	def _build(self):
		# Phi
		self._phi = [
			GraphConvolution((6, 32), act=tf.nn.relu, name='phi0'),
			GraphConvolution((32, 64), act=tf.nn.relu, name='phi1'),
			GraphConvolution((64, 128), act=tf.nn.relu, name='phi2'),
			GraphConvolution((128, 256), act=tf.nn.relu, name='phi3'),
		]
		# Global
		self._glb = [
			FullyConnected((256, 256), act=tf.nn.relu, name='glb0'),
			MaxPool()
		]
		# Omega
		self._omega = [
			FullyConnected((518, 128), act=tf.nn.relu, name='omega0'),
			FullyConnected((128, 64), act=tf.nn.relu, name='omega1'),
			FullyConnected((64, 32), act=tf.nn.relu, name='omega2'),
			FullyConnected((32, 24), act=tf.nn.relu, name='omega3')
		]
		# Psi
		self._psi = [
			FullyConnected((518, 256), act=tf.nn.relu, name='psi0'),
			FullyConnected((256, 256), act=tf.nn.relu, name='psi1'),
			FullyConnected((256, 256), act=tf.nn.relu, name='psi2'),
			FullyConnected((256, self._psd_dim * 3), name='psi3')
		]
		# chi
		self._chi = [
			FullyConnected((518, 256), act=tf.nn.relu, name='chi0'),
			FullyConnected((256, 256), act=tf.nn.relu, name='chi1'),
			FullyConnected((256, 256), act=tf.nn.relu, name='chi2'),
			FullyConnected((256, self._psd_dim * 3), name='chi3')
		]
		# Pose embedding
		self._mlp = [
			FullyConnected((72, 256), act=tf.nn.relu, name='fc0'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc1'),
			FullyConnected((256, 256), act=tf.nn.relu, name='fc2'),
			FullyConnected((256, self._psd_dim), act=tf.nn.relu, name='fc3'),
		]
		
	# Returns list of model variables
	def gather(self, only_chi=False):
		vars = []
		if only_chi: components = self._chi
		else: components = self._phi + self._glb + self._omega + self._psi + self._chi + self._mlp
		for l in components:
			vars += l.gather()
		return vars
	
	# loads pre-trained model (checks differences between model and checkpoint)
	def load(self, checkpoint):
		# checkpoint: path to pre-trained model
		# list vars
		vars = self.gather()
		# load vars values
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		values = np.load(checkpoint, allow_pickle=True)[()]
		# assign
		_vars = set([v.name for v in vars])
		_vars_chck = set(values.keys()) - {'best'}
		_diff = sorted(list(_vars_chck - _vars))
		if len(_diff):
			print("Model missing vars:")
			for v in _diff: print("\t" + v)
		_diff = sorted(list(_vars - _vars_chck))
		if len(_diff):
			print("Checkpoint missing vars:")
			for v in _diff: print("\t" + v)
		for v in vars: 
			try: v.assign(values[v.name])
			except:
				if v.name not in values: continue
				else: 
					print("Mismatch in variable shape:")
					print("\t" + v.name)
		if 'best' in values: self._best = values['best']
		
	def save(self, checkpoint):
		# checkpoint: path to pre-trained model
		print("\tSaving checkpoint: " + checkpoint)
		# get vars values
		values = {v.name: v.numpy() for v in self.gather()}
		if self._best is not float('inf'): values['best'] = self._best
		# save weights
		if not checkpoint.endswith('.npy'): checkpoint += '.npy'
		np.save(checkpoint, values)
	
	# Computes geometric descriptors for each vertex of the template outfit
	def _descriptors(self, X, L):
		# X: template outfit verts
		# L: template outfit laplacian
		for l in self._phi: X = l(X, L)
		return X
		
	# computes global descriptor
	def _global(self, X, indices):
		for l in self._glb:
			if l.__class__ == MaxPool:
				X = l(X, indices)
			else:
				X = l(X)
		return X
			
	# Computes blend weights for each descriptor of the template outfit
	def _weights(self, X):
		# X: template outfit descriptors
		for l in self._omega: X = l(X)
		# normalize weights to sum 1
		X = X / (tf.reduce_sum(X, axis=-1, keepdims=True) + 1e-7)
		return X

	# computes PSD matrices for each descriptor	
	def _psd(self, X):
		for l in self._psi: X = l(X)
		return tf.reshape(X, (-1, self._psd_dim, 3))
	
	# computes PSD matrices for each descriptor	
	def _psd_chi(self, X):
		for l in self._chi: X = l(X)
		return tf.reshape(X, (-1, self._psd_dim, 3))
	
	# computes high-level pose embedding
	def _embedding(self, X):
		for l in self._mlp: X = l(X)
		X /= tf.reduce_sum(X, axis=-1, keepdims=True)
		return X
		
	# combines pose embedding and PSD matrices to obtain final deformations
	def _deformations(self, X, PSD, indices):
		D = []
		for i in range(1, len(indices)):
			s, e = indices[i - 1], indices[i]
			_X = X[i - 1]
			_PSD = PSD[s:e]
			_D = tf.einsum('a,bac->bc', _X, _PSD)
			D += [_D]
		return tf.concat(D, axis=0)

	# Concat local and global descriptors (N x F0 & B x F1; N=verts, B=batch)	
	def _concat_descriptors(self, X, F, indices):
		F_tile = []
		for i in range(1, len(indices)):
			n = indices[i] - indices[i - 1]
			F_tile += [tf.tile(tf.expand_dims(F[i - 1], 0), [n, 1])]
		F_tile = tf.concat(F_tile, axis=0)
		return tf.concat((X, F_tile), axis=-1)
	
	# Computes the skinning for each outfit/pose
	def _skinning(self, T, W, G, indices):
		V = []
		for i in range(1, len(indices)):
			s, e = indices[i - 1], indices[i]
			_T = T[s:e]
			_G = G[i - 1]
			_weights = W[s:e]
			_G = tf.einsum('ab,bcd->acd', _weights, _G)
			_T = tf.concat((_T, self._ones(tf.shape(_T))), axis=-1)
			_T = tf.linalg.matmul(_G, _T[:,:,None])[:,:3,0]
			V += [_T]
		return tf.concat(V, axis=0)
	
	# Computes the transformation matrices of each joint of the skeleton for each pose
	def _transforms(self, poses, shapes, genders, with_body):
		G = []
		B = []
		for p,s,g in zip(poses, shapes, genders):
			_G, _B = self.SMPL[g].set_params(pose=p, beta=s, with_body=with_body)
			G += [_G]
			B += [_B]
		return np.stack(G), np.stack(B)
	
	def _ones(self, T_shape):
		return tf.ones((T_shape[0], 1), tf.float32)
	
	def __call__(self, T, L, P, S, G, fabric, tightness, indices, with_body=False):
		# T: template outfits
		# L: laplacian
		# P: poses
		# S: shapes
		# G: genders
		# fabric: fabric as per-vertex one-hot encoding
		# tightness: 2-dimensional array as described in CLOTH3D paper
		# indices: splits among outfits (multiple outfits per batch)
		# with_body: compute posed SMPL body
		
		""" STATIC """
		# local descriptors
		X = self._descriptors(T, L)
		# global descriptors
		GLB = self._global(X, indices)
		# final descriptors
		X = self._concat_descriptors(X, GLB, indices)
		X = tf.concat((X, fabric), axis=-1)
		X = self._concat_descriptors(X, tightness, indices)
		# weights
		self.W = self._weights(X)
		# psd
		PSD = self._psd(X)
		PSD_chi = self._psd_chi(tf.stop_gradient(X))
		""" DYNAMIC """
		X = self._embedding(P)
		# deformations
		self.D = self._deformations(X, PSD, indices)
		self.D_chi = self._deformations(tf.stop_gradient(X), PSD_chi, indices)
		# Compute forward kinematics and skinning
		Gs, B = self._transforms(P, S, G, with_body)
		V = self._skinning(T[:,:3] + self.D, self.W, Gs, indices)
		V_chi = self._skinning(T[:,:3] + tf.stop_gradient(self.D) + self.D_chi, tf.stop_gradient(self.W), Gs, indices)
		return V, V_chi, B
