import glob
import os

import numpy as np
import rasterio as rio
from core.frame_info import FrameInfo
from patchify import patchify, unpatchify
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels, watershed
from tensorflow.keras.metrics import BinaryIoU
from tqdm.notebook import tqdm


def get_frames(im_dirs, config):
    """Get list of frames from designated image directories.

    Args:
        im_dirs (list): List of image directories holding images, labels, and weights
        config (object of class Config): Config object containing directory names for
        RGBI, NDVI, labels, and weights directories.

    Returns:
        list: List of frames
    """
    frames = []
    for d in tqdm(im_dirs):
        rgbi_im = rio.open(glob.glob(os.path.join(d, config.rgbi_dn, "*.tif"))[0])
        ndvi_im = rio.open(glob.glob(os.path.join(d, config.ndvi_dn, "*.tif"))[0])
        label_im = rio.open(glob.glob(os.path.join(d, config.label_dn, "*.tif"))[0])
        weights_im = rio.open(glob.glob(os.path.join(d, config.weights_dn, "*.tif"))[0])

        read_rgbi_im = (np.moveaxis(rgbi_im.read(), 0, -1)) / 255  # Scale to 0-1
        read_ndvi_im = (np.moveaxis(ndvi_im.read(), 0, -1) + 1) / 2  # Scale to 0-1
        read_label_im = np.moveaxis(label_im.read(), 0, -1)
        read_weights_im = np.moveaxis(weights_im.read(), 0, -1)

        if config.use_binary_labels:
            read_label_im[read_label_im > 0] = 1  # Binarize labels

        comb_im = np.dstack((read_rgbi_im, read_ndvi_im))
        f = FrameInfo(comb_im, read_label_im, read_weights_im, d)
        frames.append(f)

    return frames


def unpatch_to_tile(patches, frame_shape, tile_shape):
    """Takes a list of patches from frames and unpatchifies them into the original tile.

    Args:
        patches (ndarray): Array of patches from the frames (h, w, n)
        frame_shape (tup): Shape of the original frames (h, w, n)
        tile_shape (tup): Shape of original tile (h, w, n)

    Raises:
        Exception: Patch count must be divisible by frame count.

    Returns:
        ndarray: Restitched tile image
    """

    frame_count = (tile_shape[0] * tile_shape[1]) // (frame_shape[0] * frame_shape[1])
    patch_shape = patches[0].shape

    if len(patches) % frame_count != 0:
        raise Exception("Patch count must be divisible by frame count.")

    tile_patch_shape = patchify(np.ones(tile_shape), frame_shape, frame_shape[0]).shape
    frame_patch_shape = patchify(
        np.ones(frame_shape), patch_shape, patch_shape[0]
    ).shape

    # First we need to unpatchify the patches into their original frame shapes
    step = len(patches) // frame_count
    frames = []
    for i in range(frame_count):
        frame = patches[step * i : step * (i + 1), ...]
        frame = np.reshape(frame, frame_patch_shape)
        frame = unpatchify(frame, frame_shape)
        frames.append(frame)
    frames = np.array(frames)

    # Then we can unpatchify the frames into the original tile shape
    tile = np.reshape(frames, tile_patch_shape)
    tile = unpatchify(tile, tile_shape)

    return tile


def get_biou(y_pred, y_true):
    """Return the binary intersection-over-union (bIoU) of the observed and predicted
    values.

    Args:
        y_pred (ndarray): Predicted values
        y_true (ndarray): Observed values

    Returns:
        float: Binary intersection-over-union value
    """
    biou = BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
    biou.update_state(y_pred=y_pred, y_true=y_true)
    res = biou.result().numpy()
    return res


def get_trees(y_pred, min_dist=1):
    """Locates individual tree labels via watershed segmentation of a binary prediction
    image.

    Args:
        y_pred (ndarray): Image containing binary predictions

    Returns:
        ndarray: Image containing segmented trees
    """
    y_pred = np.squeeze(y_pred).astype(int)
    distance = ndi.distance_transform_edt(y_pred)
    coords = peak_local_max(distance, min_distance=min_dist)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=y_pred)
    labels = label(labels)
    regions = regionprops(labels)
    return labels, regions


def chull(labels):
    """Reshapes all labels into their convex hull

    Returns:
        ndarray: New labels re-shaped by their convex hulls
    """
    chulls = np.zeros_like(labels)
    labels = label(labels)
    regions = regionprops(labels)

    for i in range(1, labels.max()):
        coords = regions[i].slice
        chulls[coords] = np.where(regions[i].image_convex, i, chulls[coords])

    labels = label(chulls)
    regions = regionprops(labels)

    return labels, regions


def grow(labels, dist=7):
    # dilated = np.zeros_like(labels)
    # for i in tqdm(range(1, labels.max()), desc="Dilating"):
    #     l = binary_dilation(labels == i, disk(radius))
    #     dilated[l] = i

    # labels = label(dilated)
    # regions = regionprops(labels)
    labels = expand_labels(labels, dist)
    regions = regionprops(labels)

    return labels, regions


def mask_bg(im):
    im = np.ma.masked_where(im == 0, im)
    return im


def write_tiff(fn, im, profile="label"):
    if profile == "label":
        profile_dict = {
            "driver": "GTiff",
            "dtype": "int16",
            "nodata": 0,
            "width": 4608,
            "height": 2048,
            "count": 1,
            "crs": None,
        }
    elif profile == "rgb":
        profile_dict = {
            "driver": "GTiff",
            "dtype": "int16",
            "nodata": np.nan,
            "width": 4608,
            "height": 2048,
            "count": 3,
            "crs": None,
        }
    with rio.open(fn, "w", **profile_dict) as dst:
        dst.write(im)


def sample_img(im, xmin=700, ymin=1000, step=500, random=False):
    if random:
        xmin = np.random.randint(0, im.shape[0] - step)
        ymin = np.random.randint(0, im.shape[1] - step)

    return im[xmin : xmin + step, ymin : ymin + step, ...]
