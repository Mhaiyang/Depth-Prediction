"""
  @Time    : 2018-8-29 00:00
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : mirror segmentation
  @File    : train.py
  @Function: train code.

"""
import os
import depth
import mhy.fcrn as modellib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "log", "fcrn")
    
config = depth.DepthConfig()
config.display()

# Configuration
dataset_root_path = os.path.abspath(os.path.join(ROOT_DIR, "./nyu_depth_v2"))
train_folder = dataset_root_path + "/train"
val_folder = dataset_root_path + "/val"
train_image_folder = train_folder + "/image"
train_depth_folder = train_folder + "/depth"
val_image_folder = val_folder + "/image"
val_depth_folder = val_folder + "/depth"
train_imglist = os.listdir(train_image_folder)
train_count = len(train_imglist)
val_imglist = os.listdir(val_image_folder)
val_count = len(val_imglist)
print("Train Image Count : {} \nValidation Image Count : {}".format(train_count, val_count))

# Training dataset
dataset_train = depth.DepthDataset()
dataset_train.load_info(train_count, train_image_folder,
                        train_depth_folder, train_imglist)     # add class and add image.
dataset_train.prepare("train")

# Validation dataset
dataset_val = depth.DepthDataset()
dataset_val.load_info(val_count, val_image_folder,
                      val_depth_folder, val_imglist)      # add class and add image
dataset_val.prepare("validation")

# ## Create Model  ###
model = modellib.FCRN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "resnet101"  # resnet or last

if init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# ## Training

# 1. Train the all layers 1e-2
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='all')
model_path = os.path.join(MODEL_DIR, "depth_fcrn_all_100.h5")
model.keras_model.save_weights(model_path)

# 2. Fine tune all layers 1e-3
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=150,
            layers="all", save_model_each_epoch=False)
model_path = os.path.join(MODEL_DIR, "depth_fcrn_all_150.h5")
model.keras_model.save_weights(model_path)

