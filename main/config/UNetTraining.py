import os


# Configuration parameters for 2-UNetTraining.ipynb
class Config:
    def __init__(self):
        self.image_dir = "../../data/dap05/combined/512"  # Relative to the notebook
        self.rgbi_dn = "rgbi"  # dn = directory name
        self.ndvi_dn = "ndvi"
        self.label_dn = "labels"
        self.weights_dn = "weights"

        # Whether to use binary segmentation or multiclass
        self.use_binary_labels = True

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
        self.input_image_channels = list(range(0, self.input_shape[-1]))
        self.input_label_channel = [5]
        self.input_weight_channel = [6]

        # CNN hyperparameters
        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 50