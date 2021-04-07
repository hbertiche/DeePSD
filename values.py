import numpy as np

# path to data
SRC = ''
# path to preprocessings
SRC_PREPROCESS = ''
# CLOTH3D rest pose
rest_pose = np.zeros((24,3))
rest_pose[0, 0] = np.pi / 2
rest_pose[1, 2] = .15
rest_pose[2, 2] = -.15