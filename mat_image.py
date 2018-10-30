"""
  @Time    : 2018-10-29 23:52
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : FCRN-DepthPrediction
  @File    : mat_image.py
  @Function: 
  
"""
import numpy as np
import h5py
import os
from PIL import Image

f = h5py.File("/media/taylor/mhy/Classification_Pretrained_Model/nyu_depth_v2_labeled.mat")
images = f["images"]
images = np.array(images)

path_converted = '/home/taylor/FCRN-DepthPrediction/nyu_depth_v2/image/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

images_number = []
for i in range(len(images)):
    images_number.append(images[i])
    a = np.array(images_number[i])

    r = Image.fromarray(a[0]).convert('L')
    g = Image.fromarray(a[1]).convert('L')
    b = Image.fromarray(a[2]).convert('L')
    img = Image.merge("RGB", (r, g, b))
    img = img.rotate(270)

    iconpath = path_converted + str(i)+'.jpg'
    img.save(iconpath, optimize=True)
    print(iconpath)
