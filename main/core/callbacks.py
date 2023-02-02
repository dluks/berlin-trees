import os

import tensorflow as tf


def checkpoint(model_fn, weights_only=False):
    """Configure and enable the ModelCheckpoint callback

    Args:
        model_path (str): Save path for the model and/or weights
        weights_only (bool, optional): Whether or not to save weights only. Defaults to
        False.

    Returns:
        object: Instance of the tf.keras.callbacks.ModelCheckpoint class
    """
    
    cp = tf.keras.callbacks.ModelCheckpoint(
        model_fn,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=weights_only,
    )
    return cp


def tensorboard(log_dir):
    """Configure and enable the Tensorboard callback

    Args:
        log_dir (str): Path of the log directory

    Returns:
        object: Instance of class tf.keras.callbacks.Tensorboard
    """
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        update_freq="epoch",
    )
    return tb
