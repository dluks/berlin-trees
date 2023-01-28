import os


# Configuration parameters for 2-UNetTraining.ipynb
class Config:

    def __init__(self):
        self.image_dir = "/Users/lusk/projects/uni-potsdam/internships/trees/data/dap05/combined/512"
        self.rgbi_dir_name = "rgbi"
        self.ndvi_dir_name = "ndvi"
        self.label_dir_name = "labels"
        self.weights_dir_name = "weights"