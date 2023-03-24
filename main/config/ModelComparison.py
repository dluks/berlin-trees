import core.optimizers as optimizers
from core.losses import (
    accuracy,
    dice_coef,
    dice_loss,
    sensitivity,
    specificity,
    tversky,
)


class Config:
    """Configuration constructor for comparing trained models
    """
    
    def __init__(self):
        # Location of the images on which predictions will be run. These should be
        # spatially sequential for restitching of final treemap.
        self.image_dir = "../../data/dap05/combined/512/393_5823*"
        self.rgbi_dn = "rgbi"  # dn = directory name
        self.ndvi_dn = "ndvi"
        self.label_dn = "labels"
        self.weights_dn = "weights"
        
        # Whether to use single or multiclass labels
        self.use_binary_labels = True
        
        # Shape of the input data, height*width*channel. Here channels are R, G, B, NIR,
        # NDVI, after which labels and weights will be added.
        self.input_image_channels = [0, 1, 2, 3, 4]
        self.input_label_channel = [5]
        self.input_weight_channel = [6]
        
        # Patch generation; from the training areas (extracted in the last notebook),
        # we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a
        # random location inside that training area. Uses a lazy strategy i.e. batch of
        # patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches
        # extracted from these areas sequential with a given step size. All the possible
        # patches are returned in one call.
        self.patch_generation_strategy = "sequential"  # 'random' or 'sequential'
        self.patch_size = (256, 256)  # Height * Width
        # # When strategy == sequential, then you need the step_size as well
        self.step_size = 256
        
        # CNN hyperparameters
        # self.BATCH_SIZE = 16
        # self.EPOCHS = 200
        self.optimizer = optimizers.adaDelta
        self.OPTIMIZER_NAME = "AdaDelta"
        self.loss = tversky
        self.LOSS_NAME = "weightmap_tversky"
        self.metrics = [
            dice_coef,
            dice_loss,
            specificity,
            sensitivity,
            accuracy,
        ]