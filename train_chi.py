import os
import sys
from random import shuffle
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from time import time
from datetime import timedelta
from math import floor

from Data.data import Data
from Model.DeePSDchi import DeePSD

from util import model_summary
from Losses import *

""" ARGS """
# gpu_id: GPU slot to run model
# name: name under which model checkpoints will be saved
# checkpoint: pre-trained model (must be in ./checkpoints/ folder)
gpu_id = sys.argv[1] # mandatory
name = sys.argv[2]   # mandatory
checkpoint = 'checkpoints/' + sys.argv[3] # mandatory

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 100
if name == 'test': stdout_steps = 1

""" TRAIN PARAMS """
batch_size = 4
virtual_batch_size = 1
num_epochs = 100
shuffle_poses = True # use random poses

edge_weight = 15
bend_weight = 1e-4
collision_weight = 25
regularizer = True # first epoch only, ensure 'chi' deformations are small

""" MODEL """
print("Building model...")
model = DeePSD(128, checkpoint)
tgts = model.gather(only_chi=True) # model weights
model_summary(tgts)
print("*"*25)
print("Model Best: ", model._best)
print("*"*25)
optimizer = tf.optimizers.Adam()

""" DATA """
print("Reading data...")
tr_txt = 'Data/train.txt'
val_txt = 'Data/val.txt'
tr_data = Data(tr_txt, batch_size=batch_size)
val_data = Data(val_txt, batch_size=batch_size)

tr_steps = floor(len(tr_data._samples) / batch_size)
val_steps = floor(len(val_data._samples) / batch_size)
for epoch in range(num_epochs):
	if (epoch + 1) % 2 == 0 and virtual_batch_size < 32: virtual_batch_size *= 2
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	total_time = 0
	metrics = [0] * 4 # Err, Edge, Bend, Collision
	cgrds = None
	start = time()
	for step in range(tr_steps):
		""" I/O """
		batch = tr_data.next()
		_idx = list(range(batch_size))
		if shuffle_poses: shuffle(_idx)
		""" Train step """
		with tf.GradientTape() as tape:
			pred, pred_chi, body = model(
						batch['template'],
						batch['laplacians'],
						batch['poses'][_idx],
						batch['shapes'],
						batch['genders'],
						batch['fabric'],
						batch['tightness'],
						batch['indices'],
						with_body=True
					)
			# Losses & Metrics
			if not shuffle_poses:
				_, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
			else:
				E_L2 = tf.constant(0.0)
			L_edge, E_edge = edge_loss(pred_chi, batch['template'][:,:3], batch['edges'])
			L_bend, E_bend = bend_loss(pred_chi, batch['faces'], batch['laplacians'])
			L_collision, E_collision, _ = collision_loss(pred_chi, body, model.SMPL[0].faces, batch['indices'])
			loss = edge_weight * L_edge \
					+ bend_weight * L_bend \
					+ collision_weight * L_collision
			# priors (D_chi should be small)
			if epoch == 0 and regularizer:
				ww = 10 ** (1 - step / tr_steps)
			else:
				ww = .001
			loss += ww * tf.reduce_sum(model.D_chi ** 2)
		""" Backprop """
		grads = tape.gradient(loss, tgts)
		if virtual_batch_size is not None:
			if cgrds is None: cgrds = grads
			else: cgrds = [c + g for c,g in zip(cgrds,grads)]
			if (step + 1) % virtual_batch_size == 0:
				optimizer.apply_gradients(zip(cgrds, tgts))
				cgrds = None
		else:
			optimizer.apply_gradients(zip(grads, tgts))
		""" Progress """
		metrics[0] += E_L2.numpy() 
		metrics[1] += E_edge.numpy()
		metrics[2] += E_bend.numpy()
		metrics[3] += E_collision
		total_time = time() - start
		ETA = (tr_steps - step - 1) * (total_time / (1+step))
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(tr_steps) + ' ... '
					+ 'Err: {:.2f}'.format(1000 * metrics[0] / (1+step)) 
					+ ' - '
					+ 'E: {:.2f}'.format(1000 * metrics[1] / (1+step))
					+ ' - '
					+ 'B: {:.3f}'.format(metrics[2] / (1+step))
					+ ' - '
					+ 'C: {:.4f}'.format(metrics[3] / (1+step))
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
	""" Epoch results """
	metrics = [m / (step + 1) for m in metrics]
	print("")
	print("Total error: {:.5f}".format(1000 * metrics[0]))
	print("Total edge: {:.5f}".format(1000 * metrics[1]))
	print("Total bend: {:.5f}".format(metrics[2]))
	print("Total collision: {:.5f}".format(metrics[3]))
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" VALIDATION """
	print("Validating...")
	total_time = 0
	metrics = [0] * 5 # Err, Err_chi, Edge, Bend, Collision
	start = time()
	for step in range(val_steps):
		""" I/O """
		batch = val_data.next()
		""" Forward pass """
		pred, pred_chi, body = model(
					batch['template'],
					batch['laplacians'],
					batch['poses'],
					batch['shapes'],
					batch['genders'],
					batch['fabric'],
					batch['tightness'],
					batch['indices'],
					with_body=True
				)
		""" Metrics """
		_, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
		_, E_L2_chi = L2_loss(pred_chi, batch['vertices'], batch['indices'])
		_, E_edge = edge_loss(pred_chi, batch['template'][:,:3], batch['edges'])
		_, E_bend = bend_loss(pred_chi, batch['faces'], batch['laplacians'])
		_, E_collision, _ = collision_loss(pred_chi, body, model.SMPL[0].faces, batch['indices'])
		""" Progress """
		metrics[0] += E_L2.numpy()
		metrics[1] += E_L2_chi.numpy()
		metrics[2] += E_edge.numpy()
		metrics[3] += E_bend.numpy()
		metrics[4] += E_collision
		total_time = time() - start
		ETA = (val_steps - step - 1) * (total_time / (1+step))	
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(val_steps) + ' ... '
					+ 'Err: {:.2f}'.format(1000 * metrics[0] / (1+step)) 
					+ ' - '
					+ 'ErrChi: {:.2f}'.format(1000 * metrics[1] / (1+step)) 
					+ ' - '
					+ 'E: {:.2f}'.format(1000 * metrics[2] / (1+step))
					+ ' - '
					+ 'B: {:.3f}'.format(metrics[3] / (1+step))
					+ ' - '
					+ 'C: {:.4f}'.format(metrics[4] / (1+step))
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
	""" Epoch results """
	metrics = [m / (step + 1) for m in metrics]
	print("")
	print("Total error: {:.5f}".format(1000 * metrics[0]))
	print("Total error chi: {:.5f}".format(1000 * metrics[1]))
	print("Total edge: {:.5f}".format(1000 * metrics[2]))
	print("Total bend: {:.5f}".format(metrics[3]))
	print("Total collision: {:.5f}".format(metrics[4]))
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" Save checkpoint """
	if metrics[0] < model._best: 
		model._best = metrics[0]
	model.save('checkpoints/' + name + str(epoch))
	print("")
	print("BEST: ", model._best)
	print("")