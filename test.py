"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import numpy as np
import skimage.io
import skimage.measure
import mhy.visualize as visualize
from depth import DepthConfig
# Important, need change when test different models.
import mhy.fcrn as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "fcrn")
DEPTH_MODEL_PATH = os.path.join(MODEL_DIR, "depth_fcrn_all_150.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "nyu_depth_v2", "test", "image")
DEPTH_DIR = os.path.join(ROOT_DIR, "nyu_depth_v2", "test", "depth")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'nyu_depth_v2', 'test', "output_fcrn_150")
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

PSNR = []
SSIM = []

for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    gt_depth = skimage.io.imread(os.path.join(DEPTH_DIR, imgname[:-4]+".png"))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    # Save results
    predict_depth = r["depth"][0, :, :, 0]
    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################
    skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname[:-4]+"_depth.png"), (predict_depth).astype(np.uint8))

    psnr = skimage.measure.compare_psnr(gt_depth, predict_depth)
    ssim = skimage.measure.compare_ssim(gt_depth, predict_depth)

    print("psnr : {}".format(psnr))
    print("ssim : {}".format(ssim))
    PSNR.append(psnr)
    SSIM.append(ssim)

mean_psnr = sum(PSNR)/len(PSNR)
mean_ssim = sum(SSIM)/len(SSIM)

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f}".format("mean_psnr", mean_psnr,
                                                                "mean_ssim", mean_ssim))





