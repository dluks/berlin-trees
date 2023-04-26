import numpy as np
from core.utils import get_biou, get_trees
from skimage.measure import label, regionprops


class Prediction:
    def __init__(self, img, model, y_true, y_true_eroded):
        y_true_bin = np.where(y_true >= 1, 1, 0)
        y_true_eroded_bin = np.where(y_true_eroded >= 1, 1, 0)
        self.img = img.astype(np.int16)
        self.model = model
        self.trained_on = "eroded" if "eroded" in self.model else "non-eroded"
        # Set border weight types
        if "brd" in self.model:
            self.weights = "border"
        elif "bounds" in self.model:
            self.weights = "bounds"
        elif "no-weights" in self.model:
            self.weights = "no weights"
            self.wt_scheme = "all1"
        else:
            self.weights = "ronneberger"
            self.wt_scheme = self.weights
        # Set border weight schemes
        if "obj" in self.model:
            self.wt_scheme = self.model.split("_")[-3]
        elif "no-weights" not in self.model and "obj" not in self.model:
            self.wt_scheme = "continuous0-10"
        self.biou_uneroded = get_biou(self.img, y_true_bin)
        self.biou_eroded = get_biou(self.img, y_true_eroded_bin)
        self.trees = get_trees(self.img, min_dist=9)
        self.trees = label(self.trees)
        self.regions = regionprops(self.trees)
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
