import numpy as np
import cv2
import natsort

from pathlib import Path


class DataSet:
    """
    Class holding functions for loading and preprocessing data.
    """
    @staticmethod
    def grain_rgb(array: np.ndarray, size: int) -> np.ndarray:
        """Dimension reduction by integration over (global) array axis, resizing, rescaling and stacking.
        From a 3rd order tensor (input) to 3 matrices. Each matrix is treated as  a single channel of an RGB image.
        Values are rescaled into range 0-255.

        Args:
            array (np.ndarray): 3rd order tensor of single grain. 0 --> matrix voxel, 1 --> garnet voxel.
            size (int): Size parameter "n" for the  n x n-matrices of the output RGB image.

        Returns:
            rgb (np.ndarray): n x n x 3-array, corresponding to an RGB image.
        """
        # sum over a axis to get a 2D matrix, then resize
        r = np.sum(array, axis=(0)).astype(np.float32)
        r = cv2.resize(r, dsize=(size, size), interpolation=cv2.INTER_LINEAR_EXACT)
        g = np.sum(array, axis=(1)).astype(np.float32)
        g = cv2.resize(g, dsize=(size, size), interpolation=cv2.INTER_LINEAR_EXACT)
        b = np.sum(array, axis=(2)).astype(np.float32)
        b = cv2.resize(b, dsize=(size, size), interpolation=cv2.INTER_LINEAR_EXACT)

        # stack the 3 2D matrices as 3 channel RGB
        rgb = np.dstack((r, g, b))
        # rescale values from 0-255
        rgb *= 255./rgb.max()

        return rgb

    """
    Load raw data (arr##.npy) and metadata for classfication and analysis.
    """
    @staticmethod
    def load_data(arr_dir: Path | str, img_size: int = 64):
        files = [a.name for a in Path(arr_dir).iterdir() if a.name.startswith('arr')]
        files = natsort.natsorted(files)

        # load data and convert to rgb img
        rgb_imgs = [DataSet.grain_rgb(np.load(Path(arr_dir, arr)), img_size) for arr in files]
        rgb_imgs = np.array(rgb_imgs)

        return rgb_imgs

    @staticmethod
    def load_metadata(arr_dir: Path | str, metadata_names: list = ["centroids", "convex_vol", "scan_dim", "voxel_counts"]):
        metadata_arrays = []
        for file_name in metadata_names:
            arr = np.load(Path(arr_dir, file_name + ".npy"))
            metadata_arrays.append(arr)

        metadata = dict(zip(metadata_names, metadata_arrays))

        return metadata

    """
    Load labelled datasets
    """
    @staticmethod
    def read_classes_from_labelled_set(dataset: Path | str, start_idx_arr_files: int = 1, sorted: bool = True):
        """Reads classes from a human-labelled dataset (with garnet explorer).

        Args:
            dataset (Path | str): Directory of the dataset. Must follow the structure:
                dataset
                ├── class1
                │   ├── arrXXX.npy
                │   ├── arrXXX.npy
                │   └── ...
                ├── class2
                │   ├── arrXXX.npy
                |   ├── ...
                └── ...

            start_idx_arr_files (int, optional): Number in file name of first arr.npy file. Defaults to 1.
            sorted (bool, optional): If True, returns classes sorted by idx. Defaults to True.

        Returns:
            human_classified_classes (np.ndarray): Array of strings with class names.
            idx_human_classified_grains (np.ndarray): Only returned if sorted=False. Array of integers with indices of grains (arr.npy) in original CT-scan.
        """
        dataset = Path(dataset)
        class_dirs = [class_dir for class_dir in dataset.iterdir() if class_dir.is_dir()]

        idx_human_classified_grains = np.array([], dtype=int)
        human_classified_classes = np.array([])

        for class_dir in class_dirs:
            # class_path = Path(dataset, class_name)
            grain_idx = [int(arr_file.stem.split("arr")[-1]) for arr_file in class_dir.iterdir() if arr_file.stem.startswith("arr")]

            if start_idx_arr_files == 1:
                # decrement 1 to get from numbers in file names (starting at 1) to indices (starting at 0)
                grain_idx = np.array(grain_idx) - 1

            elif start_idx_arr_files == 0:
                grain_idx = np.array(grain_idx)

            class_list = np.repeat(class_dir.name, len(grain_idx))

            idx_human_classified_grains = np.append(idx_human_classified_grains, grain_idx)
            human_classified_classes = np.append(human_classified_classes, class_list)

        if sorted:
            # sort classes by idx
            sorted_human_classified_classes = np.empty_like(human_classified_classes)

            for idx, human_class in zip(idx_human_classified_grains, human_classified_classes):
                sorted_human_classified_classes[idx] = human_class

            return sorted_human_classified_classes

        else:
            # return unsorted classes and corresponding indices
            return human_classified_classes, idx_human_classified_grains


class Plotting():
    @staticmethod
    def MethodComesHere():
        pass
