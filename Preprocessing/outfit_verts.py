import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from values import *
from IO import readOBJ

"""
Creates TXT files with a list of garments of each samples and vertex count of each.
CLOTH3D has different files for different garments.
DeePSD is designed to work with whole outfits.
We use this metadata to move from garment-to-outfit and viceversa.
"""
	
samples = os.listdir(SRC)
for i,sample in enumerate(samples):
	print(str(i+1)+'/'+str(len(samples)) + ' - ' + sample)
	src = SRC + sample + '/'
	dst = SRC_PREPROCESS + sample + '/'
	if os.path.isfile(dst + 'outfit_verts.txt'): continue
	# list garments
	garments = [f.replace('.obj','') for f in os.listdir(src) if f.endswith('.obj')]
	if not os.path.isdir(dst): os.mkdir(dst)
	with open(dst + 'outfit_verts.txt', 'w') as f:
		n = 0
		for i,g in enumerate(garments):
			n += readOBJ(src + g + '.obj')[0].shape[0]
			f.write(g + '\t' + str(n) + '\n')