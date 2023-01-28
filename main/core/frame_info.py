import numpy as np


class FrameInfo:
    
    def __init__(self, img, labels, weights, dtype=np.float32):
        """FrameInfo constructor

        Args:
            img (ndarray): 3D array (m, n, k) containing R, G, B, NIR, and NDVI channels
            labels (ndarray): 3D array (m, n, k) containing labels (annotations). Dimensions
            must be the same as img.
            weights (ndarray): 3D array (m, n, k) containing weights for loss calculation
            dtype (dtype, optional): Datatype of the FrameInfo array. Defaults to np.float32.
        """
        self.img = img
        self.labels = labels
        self.weights = weights
        self.dtype = dtype