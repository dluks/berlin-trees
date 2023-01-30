import numpy as np
from patchify import patchify


class FrameInfo:
    
    def __init__(self, img, labels, weights, name, dtype=np.float32):
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
        self.name = name
        self.dtype = dtype
        
    
    def all_patches(self, patch_size, step_size):
        """Get and return all sequential patches from the frame.

        Args:
            patch_size (tuple(int, int)): Size of the patch (height * width)
            step_size (tuple(int, int)): Amount of overlap between each patch. Controls
            the final number of patches returned. Should be a divisor of the patch size.

        Returns:
            ndarray: 4D array of size (n * h * w * k) containing all patches
        """
        img = self.img
        labels = self.labels
        weights = self.weights
        comb = np.dstack((img, labels, weights))
        channels = comb.shape[-1]
        
        patches = patchify(comb, (*patch_size, channels), step_size).reshape(
            -1, *patch_size, channels
        )
        
        return patches
        

    def get_single_patch(self, i, j, patch_size, img_size):
        """Gets a single patch from a frame at the given location.

        Args:
            i (int): Starting location in first dimension (x axis)
            j (int): Starting location in the second dimension (y axis)
            patch_size (tuple(int, int)): Size of the patch (height * width)
            img_size (tuple(int, int)): Total size of the image from the patch is
            generated.

        Returns:
            ndarray: 3D array of size (m * n * k) containing the desired patch
        """
        i_slice = slice(i, i + img_size[0])
        j_slice = slice(j, j + img_size[1])
        
        img = self.img[i_slice, j_slice]
        labels = self.labels[i_slice, j_slice]
        weights = self.weights[i_slice, j_slice]
        
        combined = np.dstack((img, labels, weights))
        channels = combined.shape[-1]
        
        patch = np.zeros((*patch_size, channels), dtype=self.dtype)
        patch[:img_size[0], :img_size[1]] = combined
        
        return patch
        
        
        
    def random_patch(self, patch_size):
        """Generates a randomly-located patch from the frame

        Args:
            patch_size (tuple(int, int)): Size of the patch (height * width)

        Returns:
            ndarray: 3D array of size (m * n * k) containing the random patch
        """
        img_shape = self.img.shape
        
        # Set the origin to (0,0) if the image is <= to the patch size.
        if img_shape[0] <= patch_size[0]: x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        
        if img_shape[1] <= patch_size[1]: y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        
        img_size = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch = self.get_single_patch(x, y, patch_size, img_size)
        
        return img_patch