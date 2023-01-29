import json
import os

from sklearn.model_selection import train_test_split


def split_dataset(frames, frames_json, patch_dir, test_size=0.2, val_size=0.2):
    """Split the frames into training, validation, and testing sets.

    Args:
        frames (list(FrameInfo)): Listof all the frames
        frames_json (str): Filename of the JSON cache
        patch_dir (str): Path to the directory where frame_json is stored
        test_size (float, optional): Ratio of test split. Defaults to 0.2.
        val_size (float, optional): Ratio of the validation set. Defaults to 0.2.

    Returns:
        tuple: Tuple containing the split training, validation, and testing frames
    """
    if os.path.isfile(frames_json):
        print("Reading train-test split from file")
        with open(frames_json, "r") as file:
            fjson = json.load(file)
            training_frames = fjson["training_frames"]
            testing_frames = fjson["testing_frames"]
            validation_frames = fjson["validation_frames"]

    else:
        print("Creating and writing train-test split from frames")
        frames_list = list(range(len(frames)))

        # Divide into training and test set
        training_frames, testing_frames = train_test_split(
            frames_list, test_size=test_size
        )
        
        frame_split = {
            "training_frames": training_frames,
            "testing_frames": testing_frames,
            "validation_frames": validation_frames
        }
        
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        
        with open(frames_json, "w") as f:
            json.dump(frame_split, f)
        
    print("training_frames:", training_frames)
    print("validation_frames:", validation_frames)
    print("testing_frames:", testing_frames)
    return (training_frames, validation_frames, testing_frames)