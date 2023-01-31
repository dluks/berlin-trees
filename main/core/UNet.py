import tensorflow as tf


def UNet(
    input_shape,
    input_label_channel,
    layer_count=64,
    regularizers=tf.keras.regularizers.l2(0.0001),
    weight_file=None,
):
    """Constructs the U-Net model

    Args:
        input_shape (tuple(int, int, int, int)): Shape of the input in the format of
        (batch, height, width, channels)
        input_label_channel (list[int]): List of label channels, used for calculating
        the number of channels in the model output
        layer_count (int, optional): Count of kernels in the first layer. Number of
        kernels in subsequent layers grows with a fixed factor. Defaults to 64.
        regularizers (tf.keras.regularizers, optional): Regularizers to use in each
        layer. Defaults to tf.keras.regularizers.l2(0.0001).
        weight_file (str, optional): Path to the weights file. Defaults to None.
    """
    def conv_block(input, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, 3, activation="relu", padding="same")(
            input
        )
        x = tf.keras.layers.Conv2D(num_filters, 3, activation="relu", padding="same")(x)
        return x

    def encoder_block(input, num_filters):
        x = conv_block(input, num_filters)
        x = tf.keras.layers.BatchNormalization()(x)
        p = tf.keras.layers.MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(input, skip_features, num_filters):
        x = tf.keras.layers.Conv2DTranspose(
            num_filters, (2, 2), strides=2, padding="same"
        )(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Concatenate()([x, skip_features])
        x = conv_block(x, num_filters)
        return x

    input_img = tf.keras.layers.Input(input_shape[1:], name="Input")
    s1, p1 = encoder_block(input_img, 1 * layer_count)
    s2, p2 = encoder_block(p1, 2 * layer_count)
    s3, p3 = encoder_block(p2, 4 * layer_count)
    s4, p4 = encoder_block(p3, 8 * layer_count)
    b1 = conv_block(p4, 16 * layer_count)
    d1 = decoder_block(b1, s4, 8 * layer_count)
    d2 = decoder_block(d1, s3, 4 * layer_count)
    d3 = decoder_block(d2, s2, 2 * layer_count)
    d4 = decoder_block(d3, s1, 1 * layer_count)

    outputs = tf.keras.layers.Conv2D(
        len(input_label_channel),
        1,
        padding="same",
        activation="sigmoid",
        kernel_regularizer=regularizers,
    )(d4)

    model = tf.keras.models.Model(input_img, outputs, name="U-Net")
    
    if weight_file:
        model.load_weights(weight_file)
    
    model.summary()  # Print a summary
    
    return model
