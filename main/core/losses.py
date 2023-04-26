import tensorflow as tf
import tensorflow.keras.backend as K


def tversky(y_true, y_pred, alpha=0.6, beta=0.4):
    """Calculate the Tversky loss for imbalanced classes

    Args:
        y_true (tensor): Array of the ground truth data of size (m * n * 2) where the
        last axis is labels + weights
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


def accuracy(y_true, y_pred):
    """Compute accuracy

    Args:
        y_true (tensor): Ground truth of size (m * n * 2) where last axis is labels
        weights
        y_pred (tensor): Model predictions

    Returns:
        tensor: Tensor containing pixelwise accuracies
    """
    y_t = tf.expand_dims(y_true[..., 0], -1)
    # return tf.equal(tf.round)
    return tf.equal(tf.round(y_t), tf.round(y_pred))


def dice_coef(y_true, y_pred, smooth=0.0000001):
    """Compute dice coefficient

    Args:
        y_true (tensor): Ground truth of size (m * n * 2) where last axis is labels
        weights
        y_pred (tensor): Model predictions as logits
        smooth (float, optional): Smooth by value. Defaults to 1e-7.

    Returns:
        tensor: Dice coefficient
    """
    y_t = tf.expand_dims(y_true[..., 0], -1)
    intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)

    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=-1)


def dice_loss(y_true, y_pred):
    """Compute the dice loss

    Args:
        y_true (tensor): Ground truth of size (m * n * 2) where last axis is labels
        weights
        y_pred (tensor): Model predictions as logits

    Returns:
        tensor: Dice loss
    """
    return 1 - dice_coef(y_true, y_pred)


def confusion_matrix(y_true, y_pred):
    """Calculate the confusion matrix values from ground truth and predictions

    Args:
        y_true (tensor): Tensor of shape (m * n * 2) containing labels and weights
        y_pred (tensor): Tensor of probablistic predictions

    Returns:
        tp (tensor): True positives
        fp (tensor): False positives
        tn (tensor): True negatives
        fn (tensor): False negatives
    """
    y_t = tf.expand_dims(y_true[..., 0], -1)

    tp = K.round(y_t * y_pred)
    fp = K.round((1 - y_t) * y_pred)
    tn = K.round((1 - y_t) * (1 - y_pred))
    fn = K.round((y_t) * (1 - y_pred))

    return tp, fp, tn, fn


def sensitivity(y_true, y_pred):
    """Compute sensitivity (recall)

    Args:
        y_true (tensor): Tensor of shape (m * n * 2) containing labels and weights
        y_pred (tensor): Tensor of probablistic predictions

    Returns:
        float: Sensitivity (recall)
    """
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))


def specificity(y_true, y_pred):
    """Compute specificity (precision)

    Args:
        y_true (tensor): Tensor of shape (m * n * 2) containing labels and weights
        y_pred (tensor): Tensor of probablistic predictions

    Returns:
        float: Specificity (precision)
    """
    _, fp, tn, _ = confusion_matrix(y_true, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))
