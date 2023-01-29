import os


# Configuration parameters for 2-UNetTraining.ipynb
class Config:
    def __init__(self):
        self.image_dir = "../../data/dap05/combined/512"  # Relative to the notebook
        self.rgbi_dn = "rgbi"  # dn = directory name
        self.ndvi_dn = "ndvi"
        self.label_dn = "labels"
        self.weights_dn = "weights"

        # Patch generation; from the training areas (extracted in the last notebook),
        # we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a
        # random location inside that training area. Uses a lazy strategy i.e. batch of
        # patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches
        # extracted from these areas sequential with a given step size. All the possible
        # patches are returned in one call.
        self.patch_generation_strategy = "random"  # 'random' or 'sequential'
        self.patch_size = (256, 256, 5)  # Height * Width * (Input + Output) channels
        # # When strategy == sequential, then you need the step_size as well
        # step_size = (128,128)

        # The training areas are divided into training, validation and testing set. Note
        # that training area can have different sizes, so it doesn't guarantee that the
        # final generated patches (when using sequential strategy) will be in the same
        # ratio.
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # Probability with which the generated patches should be normalized 0 -> don't
        # normalize, 1 -> normalize all
        self.normalize = 0.4

        # The split of training areas into training, validation and testing set, is
        # cached in patch_dir.
        self.patch_dir = f"./patches{self.patch_size[0]}"
        self.frames_json = os.path.join(self.patch_dir, "frames_list.json")
        
        # Shape of the input data, height*width*channel. Here channels are R, G, B, NIR,
        # NDVI, after which labels and weights will be added.
        self.input_shape = (256,256,5)
        self.input_image_channel = list(range(0, 5))
        self.input_label_channel = [5]
        self.input_weight_channel = [6]
