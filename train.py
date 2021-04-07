import os
import sys
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from time import time
from datetime import timedelta
from math import floor

from Data.data import Data
from Model.DeePSD import DeePSD

from util import model_summary
from Losses import *

""" ARGS """
# gpu_id: GPU slot to run model
# name: name under which model checkpoints will be saved
# checkpoint: pre-trained model (must be in ./checkpoints/ folder)
gpu_id = sys.argv[1] # mandatory
name = sys.argv[2]   # mandatory
checkpoint = None
if len(sys.argv) > 3:
	checkpoint = 'checkpoints/' + sys.argv[3]

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" Log """
stdout_steps = 100
if name == 'test': stdout_steps = 1 # show progress continuously (instead of every 100 steps)

""" TRAIN PARAMS """
batch_size = 4
virtual_batch_size = 1
num_epochs = 10 if checkpoint is None else 4 # if fine-tuning, fewer epochs

""" MODEL """
print("Building model...")
model = DeePSD(128, checkpoint)
if checkpoint is not None and checkpoint.split('/')[1] != name: model._best = float('inf')
tgts = model.gather() # model weights
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
	if (epoch + 1) % 2 == 0: virtual_batch_size *= 2
	print("")
	print("Epoch " + str(epoch + 1))
	print("--------------------------")
	""" TRAIN """
	print("Training...")
	total_time = 0
	error = 0
	cgrds = None
	start = time()
	for step in range(tr_steps):
		""" I/O """
		batch = tr_data.next()
		""" Train step """
		with tf.GradientTape() as tape:
			pred, body = model(
						batch['template'],
						batch['laplacians'],
						batch['poses'],
						batch['shapes'],
						batch['genders'],
						batch['fabric'],
						batch['tightness'],
						batch['indices'],
						with_body=False
					)
			# Loss & Error
			L2, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
			loss = L2
			# priors (first epoch only, hastens convergence)
			if epoch == 0 and not checkpoint:
				ww = 10 ** (1 - step / tr_steps)
				loss += ww * tf.reduce_sum((model.W - batch['weights_prior']) ** 2)
				loss += ww * tf.reduce_sum(model.D ** 2)
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
		error += E_L2.numpy() 
		total_time = time() - start
		ETA = (tr_steps - step - 1) * (total_time / (1+step))
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(tr_steps) + ' ... '
					+ 'Err: {:.2f}'.format(1000 * error / (1+step))
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
	""" Epoch results """
	error /= (step + 1)
	print("")
	print("Total error: {:.5f}".format(1000 * error)) # in millimeters
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" VALIDATION """
	print("Validating...")
	total_time = 0
	error = 0
	start = time()
	for step in range(val_steps):
		""" I/O """
		batch = val_data.next()
		""" Forward pass """
		pred, body = model(
					batch['template'],
					batch['laplacians'],
					batch['poses'],
					batch['shapes'],
					batch['genders'],
					batch['fabric'],
					batch['tightness'],
					batch['indices'],
					with_body=False
				)
		""" Metrics """
		_, E_L2 = L2_loss(pred, batch['vertices'], batch['indices'])
		""" Progress """
		error += E_L2.numpy()
		total_time = time() - start
		ETA = (val_steps - step - 1) * (total_time / (1+step))	
		if (step + 1) % stdout_steps == 0:
			sys.stdout.write('\r\tStep: ' + str(step+1) + '/' + str(val_steps) + ' ... '
					+ 'Err: {:.2f}'.format(1000 * error / (1+step)) 
					+ ' ... ETA: ' + str(timedelta(seconds=ETA)))
			sys.stdout.flush()
	""" Epoch results """
	error /= (step + 1)
	print("")
	print("Total error: {:.5f}".format(1000 * error))
	print("Total time: " + str(timedelta(seconds=total_time)))
	print("")
	""" Save checkpoint """
	if error < model._best: 
		model._best = error
		model.save('checkpoints/' + name)
	print("")
	print("BEST: ", model._best)
	print("")