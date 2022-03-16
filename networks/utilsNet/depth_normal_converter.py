import numpy as np
#import matplotlib.pyplot as plt
import cv2
import errno
import os
import glob
import argparse


#Most scenes use a virtual focal length of 35.0mm. 
# For those scenes, the virtual camera intrinsics matrix is given by
'''
[fx=1050.0	0.0	    cx=479.5
 0.0	fy=1050.0	cy=269.5
 0.0	0.0	            1.0]

'''
# Some scenes in the Driving subset use a virtual focal length of 15.0mm (the directory structure describes this clearly). 
# For those scenes, the intrinsics matrix is given by
'''
[ fx=450.0	0.0	cx=479.5
   0.0	fy=450.0	cy=269.5
   0.0	0.0	         1.0]
'''
''' Here We supposed baseline is 1 pixel'''

# Make dirs
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def disparity2depth(disparity,baseline,focus_length):
	
	depth = baseline * focus_length*1.0/disparity+1e-10
	return depth
	


def pop3d(cx, cy, depth, fx, fy):
	h, w = depth.shape[:2]
	y_coord = np.arange(0, h, 1).reshape((h, 1, 1))
	y_coord = np.repeat(y_coord, repeats=w, axis=1)
	x_coord = np.arange(0, w, 1).reshape((1, w, 1))
	x_coord = np.repeat(x_coord, repeats=h, axis=0)
	coords = np.concatenate([x_coord, y_coord], axis=2)
	ppc = np.ones(coords.shape)
	ppc[..., 0] *= cx
	ppc[..., 1] *= cy
	focal = np.ones(coords.shape)
	focal[..., 0] *= fx
	focal[..., 1] *= fy
	XY = (coords - ppc) * depth / focal

	return XY

def cal_normal(XY, Z, win_sz, dep_th):

	def cal_patch(i, j, sz):
		cent_d = Z[i+sz//2, j+sz//2, 0]
		val_mask = (np.abs(Z[i:i+sz, j:j+sz, 0] - cent_d) < dep_th * cent_d) & (Z[i:i+sz, j:j+sz, 0] > 0)
		if val_mask.sum() < 10:
			return np.array([0., 0., 0.])
		comb_patch = np.concatenate([XY[i:i+sz, j:j+sz], Z[i:i+sz, j:j+sz]], axis=2)
		A = comb_patch[val_mask]
		A_t = np.transpose(A, (1, 0))
		A_tA = np.dot(A_t, A)
		try:
			n = np.dot(np.linalg.inv(A_tA), A_t).sum(axis=1, keepdims=False)
		except:
			n = np.array([0., 0., 0.])
		return n

	h, w = Z.shape[:2]
	normal = np.zeros((h-win_sz, w-win_sz, 3))
	for i in range(h-win_sz):
		for j in range(w-win_sz):
			norm_val = cal_patch(i, j, win_sz)
			normal[i, j] = norm_val
	return normal

import sys
sys.path.append("../..")
from utils.file_io import read_img,read_disp
import matplotlib.pyplot as plt
import time

if __name__ =="__main__":
	t1 = time.time()
	left_image_path = "R.png"
	right_image_path = "L.png"
	disparity_image_path  = "0006.pfm"

	left_image = read_img(left_image_path)
	right_image = read_img(right_image_path)
	gt_disparity_left = read_disp(disparity_image_path)

	camera_intrinsic= np.array([[1050.0,0.0,479.5],
 								[0.0,1050.0,269.5],
 								[0.0,0.0,1.0]])
	# camera_intrinsic= np.array([[450.0,0.0,479.5],
 	# 							[0.0,450.0,269.5],
 	# 							[0.0,0.0,1.0]])
	fx = camera_intrinsic[0][0]
	fy = camera_intrinsic[1][1]
	cx = camera_intrinsic[0][2]
	cy = camera_intrinsic[1][2]
	
	depth = disparity2depth(gt_disparity_left,baseline=1.0,focus_length=fx)
	Z = np.expand_dims(depth, axis=2)[::2, ::2]
	XY = pop3d(cx,cy,Z,fx,fy)
	normal = cal_normal(XY, Z, 7, 0.1)
	h, w = depth.shape[:2]
	normal_u = cv2.resize(normal, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
	n_div = np.linalg.norm(normal_u, axis=2, keepdims=True)+1e-10
	normal_u /= n_div
	normal_u = (normal_u + 1.) / 2.
	t2 = time.time()

	t = t2-t1
	print(t)

	plt.imshow(normal_u)
	plt.savefig("output.png")
	plt.show()