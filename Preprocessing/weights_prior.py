import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Model/smpl')
from smpl_np import SMPLModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from IO import *
from values import *

"""
Precomputes a blend weights prior for each sample based on proximity to SMPL in canonical pose.
"""

# CLOTH3D rest pose (star)
rest_pose = np.zeros((24,3))
rest_pose[0,0] = np.pi / 2
rest_pose[1,2] = .15
rest_pose[2,2] = -.15
smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/../Model/smpl/'
SMPL = {
	0: SMPLModel(smpl_path + 'model_f.pkl', rest_pose),
	1: SMPLModel(smpl_path + 'model_m.pkl', rest_pose)
}

# get sample list
samples = os.listdir(SRC)
N = len(samples)
# for each sample
for i,sample in enumerate(samples):
	print(str(i+1) + " / " + str(N))
	src = SRC + sample + '/'
	src_pre = SRC_PREPROCESS + sample + '/'
	dst = src_pre + 'weights.npy'
	if os.path.isfile(dst): continue
	# load info
	info = loadInfo(src + 'info.mat')
	shape = info['shape']
	gender = info['gender']
	# load template outfit
	T = readPC2(src_pre + 'rest.pc16', True)['V'][0]
	# compute SMPL in rest pose
	_, B = SMPL[gender].set_params(pose=rest_pose, beta=shape, with_body=True)
	# nearest neighbour
	tree = cKDTree(B)
	_, idx = tree.query(T)
	W_prior = SMPL[gender].weights[idx]
	np.save(dst, W_prior)