# Configuration parameters for 1-Preprocessing.ipynb
class Config:
    """Configuration for 1-Preprocessing.ipynb"""

    def __init__(self):
        # Reading the raw images and annotations
        self.data_dir_path = "../../data/dap05/hand"
        self.image_file_ext = ".tif"
        self.rgbi_dn = "rgbi"  # dn = subdirectory name
        self.label_dn = "label"
        self.background_label_val = 0
        self.figures_dir = "./figures"

        # For writing ndvi, eroded labels (optional), and boundary weights
        self.ndvi_dn = "ndvi"
        self.ndvi_suffix = "ndvi"
        self.eroded_suffix = "eroded"
        self.boundary_weights_dn = "border_weights"
        self.boundary_suffix = "border_weights"
