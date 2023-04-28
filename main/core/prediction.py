import numpy as np
from core.utils import get_biou, get_trees
from skimage.measure import label, regionprops


class Prediction:
    def __init__(self, img, model, y_true, y_true_eroded):
        y_true_bin = np.where(y_true >= 1, 1, 0)
        y_true_eroded_bin = np.where(y_true_eroded >= 1, 1, 0)
        self.img = img.astype(np.int16)
        self.model = model
        self.trained_on = "ERODED" if "eroded" in self.model else "ORIG"
        # Set border weight types
        if "brd" in self.model:
            self.weights = "BORD10"
            self.wt_scheme = self.weights
        elif "bounds" in self.model:
            self.weights = "BOUNDS10"
            self.wt_scheme = self.weights
        elif "no-weights" in self.model:
            self.weights = "ALL1"
            self.wt_scheme = "ALL1"
        else:
            self.weights = "RONN"
            self.wt_scheme = self.weights
        # Set border weight schemes
        # if "obj" in self.model:
        #     self.wt_scheme = self.model.split("_")[-3]
        # elif "no-weights" not in self.model and "obj" not in self.model:
        #     self.wt_scheme = "continuous0-10"
        self.biou_uneroded = get_biou(self.img, y_true_bin)
        self.biou_eroded = get_biou(self.img, y_true_eroded_bin)
        self.trees, self.regions = get_trees(self.img, min_dist=9)
        self.tree_ct = len(self.regions)

    def sample(self, xmin=700, ymin=1000, step=500, random=False):
        if random:
            xmin = np.random.randint(0, self.trees.shape[0] - step)
            ymin = np.random.randint(0, self.trees.shape[1] - step)

        return self.trees[xmin : xmin + step, ymin : ymin + step]

    def restore_trees(self):
        labels = get_trees(self.img, min_dist=9)
        labels = label(labels)
        regions = regionprops(labels)
        return labels, regions
