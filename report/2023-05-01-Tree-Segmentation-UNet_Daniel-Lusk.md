---
title: "Understanding weight maps and label manipulation in tree detection from high-resolution orthophotos with U-Net"
author: "Daniel Lusk"
author_profile: true
date: 2023-05-01
permalink: /posts/2023/05/weight-maps-label-manipulation-tree-detection-unet
toc: true
toc_sticky: true
toc_label: "Understanding weight maps and label manipulation in tree detection from high-resolution orthophotos with U-Net"
header:
  overlay_image: "https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/banner.png"
  overlay_filter: 0.3
  caption: "U-Net based tree detection using eroded labels and border weight maps"
read_time: false
tags:
  - machine learning
  - CNN
  - U-Net
  - tree detection
  - orthophotos
---

Fine individiual tree crown delineation can be achieved from RGB + NIR orthophotos using CNN-based semantic segmentation with weighted losses, but how do different weight maps perform, and how does manipulation of training label sets affect this?

# Introduction

Convolutional neural networks (CNN) have been used in vegetation remote sensing for years, and are an especially popular choice for image classification tasks [^1]$$^,$$[^2]. Fully convolutional neural networks (FCNN) such as U-Net, in particular, are currently considered best-in-class when it comes to image segmentation as their output is of the same resolution as their inputs, providing pixel-by-pixel classification, and have seen growing popularity in remote sensing image segmentation [^3]$$^,$$[^4]$$^,$$[^5]$$^,$$[^6]. Not only are FCNNs like U-Net effective at semantic segmentation—the process of classifying an image pixel-by-pixel, but not identifying distinct objects—others, such as Mask R-CNN, are capable of performing instance segmentation—the classification of not only pixels but also of objects (Figure 1) [^7]$$^,$$[^8]$$^,$$[^9]. However, while instance segmentation architectures can be powerful for tasks like tree detection, their architectures are often deep and multi-faceted and can require the optimization of a multitude of different hyperparameters. U-Net, on the other hand, is lightweight and simple to implement, with few hyperparameters, and is therefore an appealing option for researchers wishing to perform image segmentation while minimizing model tuning.

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/semantic-vs-instance.png">
  <figcaption><b>Figure 1.</b> A comparison of semantic segmentation (e.g. the output of architectures like U-Net) and instance segmentation.</figcaption>
</figure>

The problem remains, however, that U-Net can only perform semantic segmentation, i.e. it cannot provide object detection, and so additional methods are needed to process the semantically classified output into individual objects. Weight maps, for example, can be used to focus special attention on the boundaries between objects in order to train the model to be more conservative with class predictions in those regions, as has been done to achieve cell segmentation in medical images[^3]. The semantic output can then be segmented algorithmically to identify individual objects. This same approach has been applied successfully by Brandt, *et al*. (2020) and Mugabowindekwe *et al*. (2022) to remote sensing imagery for tree detection [^5]$$^,$$[^6]. To achieve this, the authors utilized weight maps corresponding to the boundaries of tree crowns during training to reduce the connectivity of neighboring trees in the semantic output and then performed several morphological operations on the resulting semantic output, such as circle fitting and region growing to recover the “missing” pixels. This ensemble process (training with weight maps + morphological post-processing) can prove difficult with imagery of dense forests in which tree crowns border each other on all sides, however, and deeper understanding of the influence of training schemes and post-processing methods may be useful in improving tree detection from optical remote sensing imagery.

In this exploration, aerial orthophotos of the city of Berlin, Germany are used in the training of seven U-Net-based models. We seek to answer three questions: i) which weight map types should be used; ii) what is the effect of training a model on eroded (shrunken) labels compared to unmodified labels; and iii) which post-processing operations are most effective in isolating individual trees from semantic output.

# Methods
## Study site and data acquisition
True aerial orthophotos of Berlin were acquired by the Berlin Office of Cartography and Geodesy in daylight hours in the summer of 2020 with a spatial resolution of 0.2 m and a positional accuracy of +/- 0.4 m [^10]$$^,$$[^11]. The data used for this study consists of red, green, blue (RGB), and near-infrared (NIR) bands, and were sourced from the TrueDOP20RGB and TrueDOP20CIR datasets available for download on Geoportal Berlin (FIS-Broker). Overall, four 1-km<sup>2</sup> tiles and one 0.5-km<sup>2</sup> tile were obtained from Geoportal Berlin for a total extent of 4.5-km<sup>2</sup> (Figure 2).

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/overview.png">
  <figcaption><b>Figure 2.</b> Overview of study area (Berlin, Germany). Training tiles are outlined in blue, and the test/validation tile is outlined in green.</figcaption>
</figure>

To generate canopy height maps (CHM) for semi-automated label generation prior to the training of the U-Net models, LiDAR point clouds acquired using airborne laser scanning (ALS) were also collected from Geoportal Berlin, with the same respective extents as the four 1-km<sup>2</sup> orthophoto tiles [^12]. The point clouds were pre-classified with the following classes: soil (class 2), low (class 3), medium (class 4), and high vegetation (class 5), outliers (class low points class 7), and default (class 0).

## Data preprocessing

*Normalization and NDVI*

All band values for RGB and NIR were scaled to 0-1, and NDVI was calculated for all images using the formula $$\frac{NIR - Red}{NIR + Red}$$. NDVI values were then also normalized to 0-1 to ensure consistency of data ranges.

## Label generation

*Training label set*

Tree labels for model training were generated for the four 1-km<sup>2</sup> tiles using in a “semi-automated” fashion. First, vegetation points were reclassified by filtering out all last-return points, as these points are most likely to be hard surfaces such as ground or buildings. From the remaining point cloud, points were further filtered out that did not fall within a maximum density threshold as informed by a k-d tree. Next, to reclaim points that had been inside vegetation but had been filtered about by the above steps, smaller point neighborhoods were again generated with the use of a k-d tree and points that fell within a now smaller threshold were reclaimed as vegetation points. After vegetation points had successfully been isolated, CHMs were generated. Tree labels were finally generated by identifying local maxima “islands” across the CHMs, applying watershed segmentation from the Python library scikit-image to the CHMs with the local maxima as markers, and filtering the resulting labels by eccentricity and total label area [^13]. These labels are referred to as “ORIG”.

It should be noted that the resulting labels, while benefiting from the advantage of being able to be generated in a matter of minutes compared to the many hours and days it would take for hand-drawn annotation at the same scale, are of lower quality than their hand-drawn counterparts, as tree shapes can be somewhat unnatural (blocky instead of smooth) and at times contained non-tree pixels.

*Validation and test label sets*

To ensure the model predictions were validated on a more reliable label set, the final 0.5-km<sup>2</sup> tile was hand-annotated using napari, a multi-dimensional image viewer for Python [^14]. Due to the limited hand-labeled dataset size, these labels were used for both validation and final model testing in lieu of additional high-quality labeled datasets (Figure 3).

In total, the training label set consisted of 7,359 trees, and the validation/test label set contained 1,193 trees.

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/hand-vs-auto-labels.png">
  <figcaption><b>Figure 3.</b> Examples of semi-automated training labels and hand-drawn validation/test labels. Note the blockiness of the training labels, as well as the small segment of building classified as a tree.</figcaption>
</figure>

*Label erosion*

To explore the effect of training the models on labels that did not include the tree canopy edges (as opposed to discouraging the learning of the borders using weighting schemes), eroded training and validation sets were generated from the semi-automated label sets. These labels are referred to as “ERODED”. Label erosion was performed with a 1x1 kernel using scikit-image’s simple morphological erosion method (Figure 4) [^13].

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/orig-vs-eroded-labels.png">
  <figcaption><b>Figure 4.</b> Non-eroded (ORIG) and eroded (ERODED) label sets.</figcaption>
</figure>

## Weight maps

Four weighting schemes were used: i) Ronneberger weight maps (RONN) as described in Ronneberger et al., (2015) [^3], computed as 

$$
\begin{aligned}
w(\mathbf{x}) = w_{c}(\mathbf{x}) + w_0 \cdot \exp \left(−\frac{ (d_1(\mathbf{x})+d_2(\mathbf{x}))^2}{2 \sigma^2}\right)
\end{aligned}
$$

and characterized by the highest weights occurring at touching (or almost touching) tree-to-tree borders with a rapid decay as tree-to-tree distances increase; ii) modified Ronneberger weight maps (BOUNDS10) in which weights >= 3 are assigned a value of 10 and weights < 1 are assigned a value of 0; iii) border weight maps (BORD10) in which the inner edge pixels of each label are set to 10 and all others to 0; and iv) no weights (ALL1) in which all “weights” are set to 1 (Figure 5).

```python
def calculate_ronneberger_weights(labels, wc=None, w0 = 10, sigma = 5):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.
    
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    
    Parameters
    ----------
    y: Numpy array
        2D array of shape (image_height, image_width) representing boolean (or binary)
        mask of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.
    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    
    # Check if mask is boolean or binary mask
    if len(np.unique(labels)) == 2:
        labels = label(labels)
        
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((labels.shape[0], labels.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        
        if wc:
            class_weights = np.zeros_like(labels)
            for k, v in wc.items():
                class_weights[labels == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(labels)
    
    return w
```

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/weight_maps.png">
  <figcaption><b>Figure 5.</b>  Comparison of weight map types. Labels have been adjusted to appear darker than the background solely for visualization purposes, and during training labels + background were set to 1. Note that the borders in BORD10 are all 10, though they may appear continuous on some displays.</figcaption>
</figure>

## Deep learning model training

*CNN architecture*

For the task of tree identification, a U-Net architecture was used with a slightly-modified structure compared to the Ronneberger, et al. structure, and batch normalization was applied before each encoding operator (downsampling) and after each decoding operator (upsampling).

*Loss function and weights*

Because a core focus of this exploration is the effect of different weight maps on prediction quality, and weight maps directly modify the losses calculated after each forward pass of the CNN, choosing an appropriate loss function can greatly affect the model’s performance. When choosing a loss function, it is important to consider whether or not there is a class imbalance in the dataset. In the Berlin trees dataset only ~28% of the pixels represent trees, and therefore can be considered an imbalanced dataset. For instances such as this where the negative class abundance significantly outweighs that of the positive class, traditional accuracy-oriented loss functions become less useful, and it is preferable to use loss functions that prioritize measuring the overlap, or intersection-over-union (IoU), of the predictions with the true values [^15]. In this case, the Tversky loss function was selected as the primary metric for training as it is similar to other popular loss functions for image segmentation such as Dice or Jaccard losses, but can be adjusted according to desired outcomes [^16].

With the Tversky loss function, specificity (proportion of false positives [FP]) and sensitivity (proportion of false negatives [FN]) can be weighted by alpha ($$\alpha$$) and beta ($$\beta$$) values, respectively, with the requirement that the sum of the two values be equal to 1. Contrary to previous approaches where FNs were weighted higher than FPs as high sensitivity is typically preferred for imbalanced datasets, the $$\alpha$$ and $$\beta$$ weights were set to 0.6 and 0.4, respectively, in order to emphasize FPs [^17]. The hypothesis here is that, as the borders of tree canopies are the focus, if the model is encouraged to be slightly more conservative when predicting a pixel is part of a tree, then the model will be less likely to predict tree crown borders with high confidence, as they are most likely to be confused with the background.

To further de-incentivize the models to classify tree crown borders as trees, all prediction groups used in the Tversky loss calculation (TPs, FPs, and FNs), are multiplied by the weight maps, increasing the influence of the predictions at the tree crown boundaries or borders on the overall loss calculation.

```python
def tversky(y_true, y_pred, alpha=0.6, beta=0.4):
  """Calculate the Tversky loss for imbalanced classes

  Args:
      y_true (tensor): Array of the ground truth data of size (m * n * 2) where the last axis is labels + weights
      y_pred (tensor): Array containing pixelwise predictions as logits
      alpha (float, optional): Weight of false positives. Defaults to 0.6.
      beta (float, optional): Weight of false negatives. Defaults to 0.4.

  Returns:
      float: Loss
  """
  # Labels
  y_t = tf.expand_dims(y_true[..., 0], -1)

  # Weights
  y_weights = tf.expand_dims(y_true[..., 1], -1)

  ones = 1
  p0 = y_pred  # Probability that pixels are class i
  p1 = ones - y_pred  # Probability that pixels are not class i
  g0 = y_t  # Ground truth
  g1 = ones - y_t

  tp = tf.reduce_sum(y_weights * p0 * g0)
  fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
  fn = beta * tf.reduce_sum(y_weights * p1 * g0)

  EPSILON = 0.00001
  numerator = tp
  denominator = tp + fp + fn + EPSILON
  score = numerator / denominator

  return 1.0 - tf.reduce_mean(score)
```

*Model training*

The input size for the U-Net models was set to 256x256 pixels, and the ratio of training to validation patches was approximately 9 to 1. In addition to the border-weighted Tversky loss function described above, the learning rate was dynamically optimized during training using the ADADELTA optimizer [^18]. Each model was trained for 200 epochs with a batch size of 16.

In total, seven models were trained using different combinations of training label type (ORIG, ERODED) and weight map (RONN, BOUNDS10, BORD10, ALL1). It should be noted that RONN + ERODED training was not performed due to initial results clearly suggesting that the other three weight maps significantly outperformed RONN training, which is why only seven models will be presented below.

## Tree segmentation from deep learning model predictions

For each trained model, predictions were performed on the full set of test data containing 1,193 trees over a 0.5-km<sup>2</sup> section of Berlin. The two-class output (0 or 1; “tree” or “not tree”) was then processed using several morphological algorithms in order to segment individual trees. First, a Euclidean distance transform was applied, after which local maxima were identified. Six minimum distances were tested to determine the optimal locations of local maxima for all models. Next, each local maximum “marker” was assigned a unique label value, and then watershed segmentation was performed using the inverted distances with the markers as the seed points.

```python
def get_trees(y_pred, min_dist=1):
  """Locates individual tree labels via watershed segmentation of a binary prediction
  image.

  Args:
      y_pred (ndarray): Image containing binary predictions

  Returns:
      ndarray: Image containing segmented trees
  """
  y_pred = np.squeeze(y_pred).astype(int)

  # Calculate the distance transform
  distance = ndi.distance_transform_edt(y_pred)

  # Get local maxima 
  coords = peak_local_max(distance, min_distance=min_dist)

  # Collect the local maxima coordinates and generate unique labels for each
  mask = np.zeros(distance.shape, dtype=bool)
  mask[tuple(coords.T)] = True
  markers, _ = ndi.label(mask)

  # Perform watershed segmentation and return resulting labels and regionprops
  labels = watershed(-distance, markers, mask=y_pred)
  labels = label(labels)
  regions = regionprops(labels)

  return labels, regions
```

At this point, the resulting labels still represented the exact extent of original output of the models—only the label values had been changed. However, because some of the models were encouraged to incorrectly identify all tree pixels, two additional morphological manipulations were applied, namely, convex hull and dilation. As some of the predicted trees were not always “tree-like” in shape, each label was reshaped into its corresponding convex hull. Next, the labels were “re-grown” (dilated) to account for the “shrunken” clusters of tree pixels that were predicted without tree crown borders by models trained on eroded labels and/or with border or boundary-excluding weight maps. For most of these morphological operations scikit-image and SciPy were used [^13], [^19].

To determine the optimal combination of minimum distance and morphological manipulation, tree count absolute error (TCAE) and binary IoU (bIoU) were calculated for all combinations, and the optimal values were used for final ensemble predictions (Figures 6 and 7). TCAE was calculated as $$\frac{\lvert y_{true} - y_{pred} \rvert}{y_{true}} \cdot 100$$.

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/min-dist_by_abs-tree-error-pct.png">
  <figcaption><b>Figure 5.</b> TCAE as minimum distance for local maxima selection increases.</figcaption>
</figure>

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/morph_vs_biou.png">
  <figcaption><b>Figure 7.</b> Effect of label set (during training) and morphological adjustment (applied to trained model predictions) on overall bIoU. Values for “no erosion” and “eroded” represent the bIoU of unmodified predictions compared to the ORIG and ERODED label sets, respectively, while “best dist chull” and “best dist dilated” refer to bIoUs of predictions that have first had watershed segmentation applied using their optimal distances and have then had convex hull and region-growing (dilation) operations applied, respectively.</figcaption>
</figure>

# Results

## Overall performance

To test the efficacy of the above methodology, in addition to assessing core model performance statistics, the influence of four key components was examined: i) whether models were trained on original or eroded label sets; ii) weight map type; iii) minimum distance threshold for local maxima calculation; and iv) convex hull and dilation. The best model/post-processing ensembles were then identified based on tree count absolute error (TCAE), tree area distributions, and overall bIoU (Table 1).

<figure>
  <table class="dataframe" style="display: inline-table;">
    <thead>
      <tr class="best-row" style="text-align: right;">
        <th>Label Set</th>
        <th>Weights</th>
        <th>Best Min-Dist</th>
        <th>Best Morph</th>
        <th>bIoU</th>
        <th>Tree Absolute Error</th>
        <th>KS-Test <em>p</em></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ORIG</td>
        <td>RONNN</td>
        <td>3</td>
        <td>dilated</td>
        <td>0.740</td>
        <td>18.52%</td>
        <td>0.639</td>
      </tr>
      <tr class="best-row">
        <td>ORIG</td>
        <td>BOUNDS10</td>
        <td>5</td>
        <td>chull</td>
        <td>0.799</td>
        <td class="best"><b>1.84%</b></td>
        <td>0.942</td>
      </tr>
      <tr>
        <td>ORIG</td>
        <td>BORD10</td>
        <td>5</td>
        <td>original</td>
        <td>0.798</td>
        <td>6.12%</td>
        <td>0.104</td>
      </tr>
      <tr class="best-row">
        <td>ORIG</td>
        <td>ALL1</td>
        <td>5</td>
        <td>chull</td>
        <td class="best"><b>0.801</b></td>
        <td>4.44%</td>
        <td>0.855</td>
      </tr>
      <tr>
        <td>ERODED</td>
        <td>BOUNDS10</td>
        <td>5</td>
        <td>chull</td>
        <td>0.777</td>
        <td>4.69%</td>
        <td>0.501</td>
      </tr>
      <tr class="best-row">
        <td>ERODED</td>
        <td>BORD10</td>
        <td>9</td>
        <td>dilated</td>
        <td>0.797</td>
        <td>2.85%</td>
        <td class="best"><b>0.954</b></td>
      </tr>
      <tr>
        <td>ERODED</td>
        <td>ALL1</td>
        <td>7</td>
        <td>chull</td>
        <td>0.781</td>
        <td>5.62%</td>
        <td>0.933</td>
      </tr>
    </tbody>
  </table>


  <figcaption><b>Table 1.</b> Core model + post-processing ensemble statistics. Best Min-Dist refers to the minimum distance used for determining local maxima prior to watershed segmentation, Best Morph is the morphological operations which provided the greatest bIoU (performed after watershed segmentation with Best Min-Dist, and bIoU, Tree [Count] Absolute Error, and KS-Test <em>p</em>-values were calculated from the resulting instance segmentation. In this case, higher KS-Test <em>p</em>-values are better as they suggest a closer relationship between the observed tree area distribution and the predicted distribution.</figcaption>
</figure>

The highest bIoU was produced by ORIG + ALL1, which was expected as there were no penalties for predictions at tree borders. ORIG + BOUNDS10 had the lowest TCAE, and ERODED + BORD10 had the highest tree area distribution similarity. The poorest performances came from ORIG + RONNN and ORIG + BORD10. In all but one case, implementing at least the convex hull operation improved bIoU, while, of the models trained on ERODED label sets only the BORD10 weight map predictions benefited from dilation. This is understandable, as pixels classified as “border” pixels are closer to the center of mass of the tree than the boundaries, therefore allowing for greater separation between tree pixel clusters than boundary pixels.

## Tree area histograms

TCAE only provides a shallow sense of quality of the instance segmentation, however, and may be indicative of the real relationship between the predicted trees and their observed counterparts. To better understand this relationship, the distribution of predicted trees by their area was compared with the observed (true) tree area distribution and Kolmogorov-Smirnov test (KS-Test) $$p$$-value significances were computed (Figure 8).

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/tree_area_hist.png">
  <figcaption><b>Figure 8.</b> Tree area distributions of predictions (“y_pred”) overlaid with the observed distribution (“y_true”). Purple indicates overlapping bars, and <em>p</em>-values indicate the possibility of rejecting the NULL hypothesis that the two histograms do not come from the same distribution.</figcaption>
</figure>

Here it can be seen that, of the highest performers, ERODED + BORD10 produces the closest-aligned output, as all other models greatly overestimate small-area trees.

## Visual inspection

Further visual inspection of the ensemble predictions suggests that this difference is perhaps more significant than the higher bIoU and lower TCAE of ORIG + BOUNDS10, as the trees of ERODED + BORD10 appears more appropriately segmented (Figures 9 and 10).

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/best_preds_abserr.png">
  <figcaption><b>Figure 9.</b> Visual plots of ensemble predictions overlaid on a sample of the original RGB image. Y_True indicates the observed test labels.</figcaption>
</figure>

<figure>
  <img src="https://github.com/UP-RS-ESP/up-rs-esp.github.io/raw/master/_posts/Tree_Segmentation_images/full_prediction.png">
  <figcaption><b>Figure 10.</b> Comparison of true labels (y_true) with full tile output of the ERODED + BORD10 ensemble.</figcaption>
</figure>

# Discussion and conclusions

Overall, the above exploration suggests that training a model on eroded labels with border (not boundary) weights may be able to produce better ensemble segmentation results than with the original label set and thresholded boundary weights, as has been done in previous tree detection attempts [^5]$$^,$$[^6]. Further, convex hull transformation (at the least) followed by label dilation can produce prediction output with nearly identical coverage of tree pixels while still allowing for accurate instance segmentation of U-Net-generated semantic segmentation output. The model trained on eroded labels with border weights and modified with both convex hull and dilation transformations was ultimately able to produce instance segmentation with a resulting bIoU only 0.004 less than the unweighted model trained on uneroded labels (0.797 compared to 0.801).

That said, this preliminary investigation could be improved in several ways. First, K-fold cross-validation was not performed during model training, and so model resilience is not reflected here. Additionally, the training labels generated semi-automatically contain many errors and some instances of unrealistic tree segmentation, and the use of the same set of higher-quality labels, all drawn from the same geographic location, for both validation and testing can be problematic when evaluating model performance as it is unlikely to result in a well-generalized model. Furthermore, the validation/test label set was generated in patches, which led to sometimes mis-matched or truncated labels when trees spanned the borders of multiple tiles. These latter issues are matters of time and labor, however, and could be resolved with investment in higher-quality label sets across broader swaths of Berlin. With these issues in mind, however, the resulting model performance remained surprisingly accurate, and it is likely that resolving them would result in even better tree detection.

 
# References
[^1]:	T. Kattenborn, J. Leitloff, F. Schiefer, and S. Hinz, “Review on Convolutional Neural Networks (CNN) in vegetation remote sensing,” ISPRS J. Photogramm. Remote Sens., vol. 173, pp. 24–49, Mar. 2021, doi: 10.1016/j.isprsjprs.2020.12.010.

[^2]:	M. Reichstein et al., “Deep learning and process understanding for data-driven Earth system science,” Nature, vol. 566, no. 7743, Art. no. 7743, Feb. 2019, doi: 10.1038/s41586-019-0912-1.

[^3]:	O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds., in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.

[^4]:	M. Perslev, E. B. Dam, A. Pai, and C. Igel, “One Network to Segment Them All: A General, Lightweight System for Accurate 3D Medical Image Segmentation,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2019, D. Shen, T. Liu, T. M. Peters, L. H. Staib, C. Essert, S. Zhou, P.-T. Yap, and A. Khan, Eds., in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2019, pp. 30–38. doi: 10.1007/978-3-030-32245-8_4.

[^5]:	M. Brandt et al., “An unexpectedly large count of trees in the West African Sahara and Sahel,” Nature, vol. 587, no. 7832, Art. no. 7832, Nov. 2020, doi: 10.1038/s41586-020-2824-5.

[^6]:	M. Mugabowindekwe et al., “Nation-wide mapping of tree-level aboveground carbon stocks in Rwanda,” Nat. Clim. Change, pp. 1–7, Dec. 2022, doi: 10.1038/s41558-022-01544-w.

[^7]:	R. Girshick, “Fast R-CNN,” presented at the Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 1440–1448. Accessed: Oct. 17, 2022. [^Online]. Available: https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html

[^8]:	S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2015. Accessed: Oct. 17, 2022. [^Online]. Available: https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html

[^9]:	K. He, G. Gkioxari, P. Dollár, and R. Girshick, “Mask R-CNN.” arXiv, Jan. 24, 2018. doi: 10.48550/arXiv.1703.06870.

[^10]:	“Geoportal Berlin / Digitale farbige TrueOrthophotos 2020 (TrueDOP20RGB) - Sommerbefliegung.” Accessed: Sep. 15, 2022. [^License: dl-de/by-2-0 (http://www.govdata.de/dl-de/by-2-0)]. Available: https://fbinter.stadt-berlin.de/fb/wms/senstadt/k_luftbild2020_true_rgb

[^11]:	“Geoportal Berlin / Digitale Color-Infrarot TrueOrthophotos 2020 (TrueDOP20CIR) - Sommerbefliegung.” Accessed: Sep. 15, 2022. [^License: dl-de/by-2-0 (http://www.govdata.de/dl-de/by-2-0)]. Available: https://fbinter.stadt-berlin.de/fb/wms/senstadt/k_luftbild2020_true_cir

[^12]:	“Geoportal Berlin / Airborne Laserscanning (ALS) - Primäre 3D Laserscan-Daten.” Accessed: Sep. 15, 2022. [^License: dl-de/by-2-0 (http://www.govdata.de/dl-de/by-2-0)]. Available: https://fbinter.stadt-berlin.de/fb/feed/senstadt/a_als

[^13]:	S. van der Walt et al., “scikit-image: image processing in Python,” PeerJ, vol. 2, p. e453, Jun. 2014, doi: 10.7717/peerj.453.

[^14]:	N. Sofroniew et al., “napari: a multi-dimensional image viewer for Python.” Zenodo, Nov. 03, 2022. doi: 10.5281/zenodo.7276432.

[^15]:	M. A. Rahman and Y. Wang, “Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation,” in Advances in Visual Computing, G. Bebis, R. Boyle, B. Parvin, D. Koracin, F. Porikli, S. Skaff, A. Entezari, J. Min, D. Iwai, A. Sadagic, C. Scheidegger, and T. Isenberg, Eds., in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2016, pp. 234–244. doi: 10.1007/978-3-319-50835-1_22.

[^16]:	S. S. M. Salehi, D. Erdogmus, and A. Gholipour, “Tversky loss function for image segmentation using 3D fully convolutional deep networks.” arXiv, Jun. 18, 2017. doi: 10.48550/arXiv.1706.05721.

[^17]:	N. Abraham and N. M. Khan, “A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation.” arXiv, Oct. 17, 2018. doi: 10.48550/arXiv.1810.07842.

[^18]:	M. D. Zeiler, “ADADELTA: An Adaptive Learning Rate Method.” arXiv, Dec. 22, 2012. doi: 10.48550/arXiv.1212.5701.

[^19]:	P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” Nat. Methods, vol. 17, no. 3, Art. no. 3, Mar. 2020, doi: 10.1038/s41592-019-0686-2.

