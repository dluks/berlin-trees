import os


# Configuration parameters for 1-Preprocessing.ipynb
class Config:

    def __init__(self):
        # Reading the raw images and annotations
        self.data_dir_path = "/Users/lusk/projects/uni-potsdam/internships/trees/data/dap05/combined/512"
        self.image_file_ext = ".tif"
        self.rgbi_dn = "rgbi"  # dn = subdirectory name
        self.label_dn = "labels"
        self.background_label_val = 0
        
        # For writing ndvi, eroded labels (optional), and boundary weights
        self.ndvi_dn = "ndvi"
        self.ndvi_suffix = "ndvi"
        self.eroded_suffix = "eroded"
        self.boundary_weights_dn = "weights"
        self.boundary_suffix = "weights"