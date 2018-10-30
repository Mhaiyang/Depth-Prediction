"""
  @Time    : 2018-10-30 18:27
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : Depth-Prediction
  @File    : debug.py
  @Function: 
  
"""
import skimage.io
import numpy as np

depth_path = "./nyu_depth_v2/image/2.jpg"
depth = skimage.io.imread(depth_path)
print(np.max(depth))
print(np.min(depth))
print(depth.dtype)