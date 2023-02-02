import numpy as np
from imgaug import augmenters as iaa


def imageAugmentationWithIAA():
    """Apply a sequence of image agumentations using imgaug

    Returns:
        object: An object of class imgaug.iaa.Sequential
    """
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),  # random crops

        # Gaussian blur and gamma contrast
        # sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),

        # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
        # sometimes(iaa.Multiply((0.75, 1.25), per_channel=True), 0.3),

        iaa.Sometimes(0.3, iaa.LinearContrast((0.3, 1.2))),
        # iaa.Add(value=(-0.5,0.5),per_channel=True),
        iaa.Sometimes(0.3, iaa.PiecewiseAffine(0.05)),
        iaa.Sometimes(0.1, iaa.PerspectiveTransform(0.01))
    ],
        random_order=True)

    return seq


class DataGenerator:
    """DataGenerator class. Defines methods for generating patches sequentially and
    randomly from given frames.
    """
    def __init__(
        self,
        input_image_channels,
        patch_size,
        frame_idx,
        frames,
        annotation_channels=[5, 6],
        augmenter=None,
    ):

        self.input_image_channels = input_image_channels
        self.patch_size = patch_size
        self.frame_idx = frame_idx
        self.frames = frames
        self.annotation_channels = annotation_channels
        self.augmenter = augmenter


    # Return all training and label images and weights (annotations), generated
    # sequentially with the given step size
    def all_sequential_patches(self, step_size):
        """Generate all patches from all assigned frames sequentially.

        Args:
            step_size (tuple(int, int)): Step size when generating patches (See
            FrameInfo.all_patches())

        Returns:
            ndarray: Image input data. Array of size (i * m * n * k) where i is the
            number of patches, m and n are height and width, and k is the number of
            input channels.
            ndarray: Labels and weights array of size (i * m * n * k) where i is the
            number of patches, m and n are height and width of the image, and k is the
            number of annotation channels.
        """
        patches = []
        for frame_id in self.frame_idx:
            frame = self.frames[frame_id]
            frame_patches = frame.all_patches(self.patch_size, step_size)
            patches.extend(frame_patches)
        
        data = np.array(patches)
        img = data[..., self.input_image_channels]
        y = data[..., self.annotation_channels]
        
        return img, y
    
    
    def random_patch(self, BATCH_SIZE):
        """Generates patches from random locations in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently)

        Returns:
            ndarray: Image input data. Array of size (i * m * n * k) where i is the
            number of patches according to the BATCH_SIZE, m and n are height and width,
            and k is the number of input channels.
            ndarray: Labels and weights array of size (i * m * n * k) where i is the
            number of patches according to the BATCH_SIZE, m and n are height and width 
            of the image, and k is the number of annotation channels.
        """
        patches = []
        for _ in range(BATCH_SIZE):
            frame_id = np.random.choice(self.frame_idx)
            frame = self.frames[frame_id]
            patch = frame.random_patch(self.patch_size)
            patches.append(patch)

        data = np.array(patches)
        
        img = data[..., self.input_image_channels]
        y = data[..., self.annotation_channels]

        return img, y
    
    
    def random_generator(self, BATCH_SIZE):
        """Generator for retrieving random patches. Yields random patches from random
        locations in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled 
            independently)

        Yields:
            X: Training data. ndarray of size (m * n * input_channels)
            y: Labels and weights. ndarray of size (m * n * 2)
        """
        if self.augmenter == "iaa":
            seq = imageAugmentationWithIAA()
        
        while True:
            X, y = self.random_patch(BATCH_SIZE)
            
            if self.augmenter == "iaa":
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                # We need to augment y as well due to operations such as crop and 
                # transform
                y = seq_det.augment_images(y)
                # Some augmentations can change the values of y, so we need to re-assign
                # values to ensure everything's the same
                labels = y[..., [0]]
                labels[labels < 0.5] = 0
                labels[labels >= 0.5] = 1
                # TODO: As it currently stands, this will overwrite the existing weight
                # scheme, which means comparing augmented training vs training without
                # augmentation won't be apples to apples due to the now different
                # weighting schemes.
                weights = y[..., [1]]
                weights[weights >= 2] = 10
                weights[weights < 2] = 1
                
                y = np.dstack((labels, weights))
                
                yield X, y
            
            else:
                labels = y[..., [0]]
                weights = y[..., [1]]
                weights[weights >= 2] = 10
                weights[weights < 2] = 1
                
                y = np.concatenate((labels, weights), axis=-1)
                
                yield X, y
                