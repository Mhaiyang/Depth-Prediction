"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import numpy as np
import skimage.io
import skimage.measure
from depth import DepthConfig
# Important, need change when test different models.
import mhy.fcrn as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "fcrn")
DEPTH_MODEL_PATH = os.path.join(MODEL_DIR, "depth_fcrn_all_150.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "nyu_depth_v2", "demo", "train")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'nyu_depth_v2', 'demo', "train_depth")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


# Configurations
class InferenceConfig(DepthConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # Up to now, batch size must be one.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.FCRN(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(DEPTH_MODEL_PATH, by_name=True)

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    height = image.shape[0]
    width = image.shape[1]
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    # Save results
    predict_depth = r["depth"][0, :, :, 0]
    if height > width:
        final_depth = predict_depth[:, 64:576]
    elif height < width:
        final_depth = predict_depth[64:576, :]

    ###########################################################################
    # ###############  Quantitative Evaluation for Single Image ###############
    ###########################################################################
    skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname[:-4]+".png"), final_depth.astype(np.uint8))





