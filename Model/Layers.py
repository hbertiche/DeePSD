import tensorflow as tf
import tensorflow_graphics as tfg

class FullyConnected:
	def __init__(self, shape, act=None, name='fc'):
		self.w = tf.Variable(tf.initializers.glorot_normal()(shape), name=name+'_w')
		self.b = tf.Variable(tf.zeros(shape[-1], dtype=tf.float32), name=name+'_b')
		self.act = act or (lambda x: x)

	def gather(self):
		return [self.w, self.b]

	def __call__(self, X):
		X = tf.einsum('ab,bc->ac', X, self.w) + self.b
		X = self.act(X)
		return X

class GraphConvolution:
	def __init__(self, shape, act=None, name='gcn'):
		self.w = tf.Variable(tf.initializers.glorot_normal()((2, *shape)), name=name+'_w')
		self.b = tf.Variable(tf.zeros(shape[-1], dtype=tf.float32), name=name+'_b')
		self.act = act or (lambda x: x)
	
	def gather(self):
		return [self.w, self.b]

	def __call__(self, X, L):
		X0 = tf.einsum('ab,bc->ac', X, self.w[0]) # Node
		X1 = tf.einsum('ab,bc->ac', X, self.w[1]) # Neigh
		# Neigh conv
		X1 = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
			X1, L, sizes=None,
			edge_function=lambda x, y: y, 
			reduction='weighted', 
			edge_function_kwargs={}
		) 
		X = X0 + X1 + self.b
		X = self.act(X)
		return X

class MaxPool:
	def gather(self):
		return []
		
	def __call__(self, X, indices):
		_X = []
		for i in range(1, len(indices)):
			s, e = indices[i - 1], indices[i]
			_X += [tf.reduce_max(X[s:e], axis=0)]
		return tf.stack(_X, axis=0)