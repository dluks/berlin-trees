import json
import os

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(
    frames,
    frames_json,
    patch_dir,
    test_size=0.2,
    val_size=0.2,
    test_override=None,
    val_override=None,
):
    """Split the frames into training, validation, and testing sets.

    Args:
        frames (list(FrameInfo)): List of all the frames
        frames_json (str): Filename of the JSON cache
        patch_dir (str): Path to the directory where frame_json is stored
        test_size (float, optional): Ratio of test split. Defaults to 0.2.
        val_size (float, optional): Ratio of the validation set. Defaults to 0.2.
        test_override (list[str]): A list of frame names to be manually selected as test data. Only
            partial names are needed for matching.
        val_override (list[str]): A list of frame names to be manually selected as validation data.
            Only partial names are needed for matching.

    Returns:
        tuple: Tuple containing the split training, validation, and testing frames
    """
    if os.path.isfile(frames_json):
        print(f"Reading train-test split from file ({os.path.basename(frames_json)})")
        with open(frames_json, "r") as file:
            fjson = json.load(file)
            training_frame_idx = fjson["training_frame_idx"]
            testing_frame_idx = fjson["testing_frame_idx"]
            validation_frame_idx = fjson["validation_frame_idx"]

    else:
        print("Creating and writing train-test split from frames...")
        frame_idx = np.arange(len(frames))
        mask = np.ones_like(frame_idx, dtype=bool)
        f_names = [os.path.basename(f.name) for f in frames]

        if test_override:
            testing_frame_idx = []
            for val in test_override:
                for j, name in enumerate(f_names):
                    if val in name:
                        testing_frame_idx.append(j)

            mask[testing_frame_idx] = False

        if val_override:
            validation_frame_idx = []
            for val in val_override:
                for j, name in enumerate(f_names):
                    if val in name:
                        validation_frame_idx.append(j)
            
            mask[validation_frame_idx] = False

        # Mask the test and/or val frames if any
        frame_idx = frame_idx[mask].tolist()
        
        # Split the sets according to overrides (or lack of them). This feels
        # inefficient...
        if test_override and val_override:
            training_frame_idx = frame_idx
        elif val_override:
            training_frame_idx, testing_frame_idx = train_test_split(
                frame_idx, test_size=test_size
            )
        elif test_override:
            training_frame_idx, validation_frame_idx = train_test_split(
                frame_idx, test_size=val_size
            )
        else:
            training_frame_idx, testing_frame_idx = train_test_split(
                frame_idx, test_size=test_size
            )
            training_frame_idx, validation_frame_idx = train_test_split(
                training_frame_idx, test_size=val_size
            )

        frame_split = {
            "training_frame_idx": training_frame_idx,
            "testing_frame_idx": testing_frame_idx,
            "validation_frame_idx": validation_frame_idx,
        }

        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)

        with open(frames_json, "w") as f:
            json.dump(frame_split, f)
        
    # Sanity check for set sizes
    if test_override:
        print(f"\nUsing test override:", test_override)
    if val_override:
        print(f"\nUsing val override:", val_override)
    print("\nTraining set size:", len(training_frame_idx))
    print("Validation set size:", len(validation_frame_idx))
    print("Testing set size:", len(testing_frame_idx))

    return training_frame_idx, validation_frame_idx, testing_frame_idx
