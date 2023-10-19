import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
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


class Classification():

    @staticmethod
    def load_model(modelh5: str, parent_dir: str | Path = Path("saved_models"), include_rescaling_layer: bool = True):
        model = keras.models.load_model(Path(parent_dir, modelh5))

        if include_rescaling_layer:
            inputs = keras.Input(shape=(64, 64, 3))
            x = keras.layers.Rescaling(scale=1./255)(inputs)
            outputs = model(x)

            model = keras.Model(inputs=inputs, outputs=outputs)

        return model


class Plotting():

    @staticmethod
    def projection(centroids: np.ndarray, scan_dim: np.ndarray, vx_cts: np.ndarray,
                   shape_classes: np.ndarray, res_in_mm: float, shapes_of_interest: list,
                   x: str, y: str, color_shape_classes: list, ax: plt.Axes,):
        centroids_mm = centroids * res_in_mm
        dim_mm = scan_dim * res_in_mm

        COORD_DICT = {"X": 1, "Y": 2, "Z": 0}

        x_margin = dim_mm[COORD_DICT[x]] * 0.1
        y_margin = dim_mm[COORD_DICT[y]] * 0.1

        for shape_class, color_class in zip(shapes_of_interest, color_shape_classes):
            sns.scatterplot(x=centroids_mm[:, COORD_DICT[x]][shape_classes == shape_class], y=centroids_mm[:, COORD_DICT[y]][shape_classes == shape_class],
                            hue=vx_cts[shape_classes == shape_class], hue_norm=(vx_cts.min(), vx_cts.max()),
                            s=0.8*((3*res_in_mm/4*np.pi) * vx_cts[shape_classes == shape_class])**(1/3),
                            palette=sns.color_palette("blend:"+color_class[0]+","+color_class[1], as_cmap=True),
                            label=shape_class, ax=ax)

        ax.set_xlim(0 - x_margin, dim_mm[COORD_DICT[x]] + x_margin)
        ax.set_ylim(0 - y_margin, dim_mm[COORD_DICT[y]] + y_margin)

        # ax.set_xticks(np.linspace(0, dim_mm[coord_dict[x]], 5))

        ax.set_xlabel(f"{x}-coordinate [mm]")
        ax.set_ylabel(f"{y}-coordinate [mm]")

        ax.set_xticks(np.arange(0, dim_mm[COORD_DICT[x]] + 0.5, 5))
        ax.set_yticks(np.arange(0, dim_mm[COORD_DICT[y]] + 0.5, 5))

        ax.set_frame_on(False)
        ax.set_aspect("equal")

        ax.legend(bbox_to_anchor=(1.0, 0.95), frameon=False)

    @staticmethod
    def crystal_size_distribtuion(vx_cts: np.ndarray, shape_classes: np.ndarray,
                                  res_in_mm: float, shapes_of_interest: list,
                                  ax: plt.Axes, color_all: str, color_shape_classes: list,
                                  log: bool = False, bins: int | None = None,):

        # calculate grain radii assuming spherical grains
        radius_vx = (((3/4)*vx_cts)/np.pi)**(1/3)
        # convert to mm
        radius = radius_vx * res_in_mm

        if log:
            radius = np.log10(radius)

        bin_range = (np.min(radius), np.max(radius))

        # set number of bins
        if bins is None:
            # set bin-width according to Freedman-Diaconis rule
            iqr = np.subtract(*np.percentile(radius, [75, 25]))
            bin_width = (2*iqr)/(len(radius)**(1/3))
            print(bin_width)

            bins = int(np.subtract(np.max(radius), np.min(radius)) / bin_width)

        # plot histogram
        sns.histplot(radius, bins=bins, ax=ax, label="total", color=color_all)

        # combine different shapes into pandas dataframe
        radii_with_shapes = pd.DataFrame({"radius": radius, "shape": shape_classes})

        radii_with_shapes.loc[~radii_with_shapes["shape"].isin(shapes_of_interest), "shape"] = None

        sns.histplot(radii_with_shapes, x="radius", hue="shape", multiple="dodge",
                     bins=bins, binrange=bin_range, ax=ax, shrink=0.9, palette=color_shape_classes, legend=True)

        if log:
            ax.set_xlabel("log(grain radius) [log(mm)]")
        else:
            ax.set_xlabel("grain radius [mm]")
        ax.set_ylabel("counts")

    @staticmethod
    def class_fraction_in_CSD(vx_cts: np.ndarray, shape_classes: np.ndarray,
                              res_in_mm: float, shapes_of_interest: list,
                              ax: plt.Axes, color_shape_classes: list,
                              log: bool = False, bins: int | None = None):

        # calculate grain radii assuming spherical grains
        radius_vx = (((3/4)*vx_cts)/np.pi)**(1/3)
        # convert to mm
        radius = radius_vx * res_in_mm

        if log:
            radius = np.log10(radius)

        bin_range = (np.min(radius), np.max(radius))

        # set number of bins
        if bins is None:
            # set bin-width according to Freedman-Diaconis rule
            iqr = np.subtract(*np.percentile(radius, [75, 25]))
            bin_width = (2*iqr)/(len(radius)**(1/3))
            print(bin_width)

            bins = int(np.subtract(np.max(radius), np.min(radius)) / bin_width)

        else:
            bin_width = np.subtract(np.max(radius), np.min(radius)) / bins

        # calcute histogram
        hist, bin_edges = np.histogram(radius, bins=bins, range=bin_range)

        shape_class_fraction = {}

        for shape_class, color_class in zip(shapes_of_interest, color_shape_classes):
            # calculate histogram
            hist_class, _ = np.histogram(radius[shape_classes == shape_class], bins=bins, range=bin_range)
            # calculate fraction
            fraction = hist_class/hist
            # plot
            mid_bin_radius = bin_edges[:-1] + 0.5*bin_width
            ax.plot(mid_bin_radius, fraction, c=color_class, label=shape_class, lw=2, marker="o", ms=5)

            if log:
                ax.set_xlabel("log(grain radius)")
            else:
                ax.set_xlabel("grain radius [mm]")

            ax.set_ylabel("fraction")

            ax.legend()

            shape_class_fraction[shape_class] = fraction

        return shape_class_fraction, mid_bin_radius
