import matplotlib.pyplot as plt  # plotting tools


def display_images(images, input_channels, annotation_channels, titles=None):
    """Display a set of images, optionally with titles.

    Args:
        images (ndarray): Array of image tensors of size (batch * height * width *
        channels)
        input_channels (list): List of input channel ids
        annotation_channels (list): List of annotation channel ids
        titles (list, optional): List of strings of size channels minus 2 to account for
        RGB stacking. Defaults to None.
    """
    # Assume RGB is present, so combine those channels and display the rest as
    # singleband
    print(images.shape)
    cols = images.shape[-1] - 2
    rows = images.shape[0]
    titles = titles if titles is not None else [""] * cols

    _, axes = plt.subplots(rows, cols, figsize=(14, 14 * rows // cols))
    for row in range(rows):
        for i, ax in enumerate(axes[row]):
            if i == 0:
                ax.imshow(images[row, ..., :3]) # RGB
            elif i == cols - 1:
                ax.imshow(images[row, ..., i + 2])
            else:
                ax.imshow(images[row, ..., i + 2])
            
            if row == 0:
                ax.set_title(titles[i])

            ax.axis("off")
    #     axes[row][0].imshow(images[row, ..., :3])  # RGB
    #     axes[row][1].imshow(images[row, ..., 3])  # NIR
    #     axes[row][2].imshow(images[row, ..., 4])  # NDVI
    #     axes[row][3].imshow(images[row, ..., ])
            
        
        
    # for i in range(rows):
    #     for j in range(cols):
    #         plt.subplot(rows, cols, (i*cols) + j + 1)
    #         plt.axis('off')
    #         plt.imshow(images[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)

