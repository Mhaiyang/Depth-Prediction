import os
import numpy as np
from PIL import Image
from mhy.config import Config
import mhy.utils as utils
import skimage.io
import skimage.color


# Configurations
class MirrorConfig(Config):
    """Configuration for training on the mirror dataset.
    Derives from the base Config class and overrides values specific
    to the mirror dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Depth"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    IMAGE_HEIGHT = 640
    IMAGE_WIDTH = 480

    BACKBONE = "resnet101"
    Pretrained_Model_Path = "/home/iccd/Depth-Prediction/pspnet101_voc2012.h5"
    # Pretrained_Model_Path = "/root/pspnet101_voc2012.h5"

    LOSS_WEIGHTS = {
        "depth_loss": 1.
    }

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = int(1400/(GPU_COUNT*IMAGES_PER_GPU))

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = int(30/(GPU_COUNT*IMAGES_PER_GPU))

    # Learning rate
    LEARNING_RATE = 0.01


# Dataset
class DepthDataset(utils.Dataset):

    def load_info(self, count, img_folder, gt_folder, imglist):
        for i in range(count):
            filestr = imglist[i].split(".")[0]  # 10.jpg for example
            image_path = img_folder + "/" + imglist[i]
            depth_path = gt_folder + "/" + filestr + ".png"
            if not os.path.exists(depth_path):
                print("image {} has no depth map".format(filestr))
                continue
            img = Image.open(depth_path)
            width, height = img.size
            self.add_image(image_id=i, image_path=image_path, width=width, height=height,
                           depth_path=depth_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['image_path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image.astype(np.float32)

    def load_depth(self, image_id):
        """Load the specified depth and return a [H,W] Numpy array.
        """
        # Load depth [height, width]
        depth = skimage.io.imread(self.image_info[image_id]['depth_path'])

        return depth.astype(np.float32)





