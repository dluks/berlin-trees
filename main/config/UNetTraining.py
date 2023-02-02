import os
import time
from functools import reduce

import core.callbacks as callbacks
import core.optimizers as optimizers
from core.losses import (
    accuracy,
    dice_coef,
    dice_loss,
    sensitivity,
    specificity,
    tversky,
)


# Configuration parameters for 2-UNetTraining.ipynb
class Config:
    """Configuration constructor for performing U-Net training.
    """
    def __init__(self):
        self.image_dir = "../../data/dap05/combined/512"  # Relative to the notebook
        self.rgbi_dn = "rgbi"  # dn = directory name
        self.ndvi_dn = "ndvi"
        self.label_dn = "labels"
        self.weights_dn = "weights"

        # Whether to use binary segmentation or multiclass
        self.use_binary_labels = True
        
        # Describe the weights file, e.g. "obj0-bg1-bounds_cnt10" where "obj0" indicates
        # labels as having weights of 0, "bg1" -> background 1, and "bounds_cnt10" ->
        # boundaries continous weights up to 10
        self.weights_type = "obj-bg1-bounds_10"

        # Patch generation; from the training areas (extracted in the last notebook),
        # we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a
        # random location inside that training area. Uses a lazy strategy i.e. batch of
        # patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches
        # extracted from these areas sequential with a given step size. All the possible
        # patches are returned in one call.
        self.patch_generation_strategy = "random"  # 'random' or 'sequential'
        self.patch_size = (256, 256)  # Height * Width
        # # When strategy == sequential, then you need the step_size as well
        self.step_size = 128

        # The training areas are divided into training, validation and testing set. Note
        # that training area can have different sizes, so it doesn't guarantee that the
        # final generated patches (when using sequential strategy) will be in the same
        # ratio.
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # The split of training areas into training, validation and testing set, is
        # cached in patch_dir.
        self.patch_dir = f"./patches_{self.patch_size[0]}"
        self.frames_json = os.path.join(self.patch_dir, "frames_list.json")

        # Shape of the input data, height*width*channel. Here channels are R, G, B, NIR,
        # NDVI, after which labels and weights will be added.
        self.input_shape = (*self.patch_size, 5)
        self.input_image_channels = [0, 1, 2, 3, 4]
        self.input_label_channel = [5]
        self.input_weight_channel = [6]

        # CNN SETTINGS
        # Where to save the model and/or weights
        self.weights_only = 0  # 0 = False, 1 = True
        self.model_dir = "./saved_models/UNet"
        self.weights_dir = "./saved_weights/UNet"
        self.log_dir = "./logs/UNet"
        
        # CNN hyperparameters
        self.BATCH_SIZE = 8
        self.EPOCHS = 50
        self.optimizer = optimizers.adaDelta
        self.OPTIMIZER_NAME = "AdaDelta"
        self.loss = tversky
        self.LOSS_NAME = "weightmap_tversky"
        self.metrics = [dice_coef, dice_loss, specificity, sensitivity, accuracy]
        
        # Maximum number of validation images to use
        self.VAL_LIMIT = 200
        
        # Maximum number of steps_per_epoch while training
        self.MAX_TRAIN_STEPS = 1000
        
        # Create the model and weigth filenames
        timestamp = time.strftime("%Y%m%d-%H%M")
        channels = self.input_image_channels
        channels = reduce(lambda a, b: a + str(b), channels, "")

        if self.weights_only:
            if not os.path.exists(self.weights_dir):
                os.makedirs(self.weights_dir)

            self.model_fn = os.path.join(
                self.weights_dir,
                f"{timestamp}_{self.OPTIMIZER_NAME}_{self.LOSS_NAME}_{channels}_{self.input_shape[0]}.hdf5"
            )
        else:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

            self.model_fn = os.path.join(
                self.model_dir,
                f"{timestamp}_{self.OPTIMIZER_NAME}_{self.LOSS_NAME}_{channels}_{self.input_shape[0]}.h5"
            )
        
        # CNN Callbacks
        checkpoint = callbacks.checkpoint(self.model_fn)
        tensorboard = callbacks.tensorboard(self.log_dir)
        self.callbacks = [checkpoint, tensorboard]
        