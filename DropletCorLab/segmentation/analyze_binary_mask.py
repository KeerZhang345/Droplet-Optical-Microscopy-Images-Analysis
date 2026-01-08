import cv2
import glob
import os
import numpy as np
from skimage.measure import label, regionprops
from typing import Dict, List, Tuple

def filter_edge_bi_sides(edge_thresh_x: int, edge_thresh_y: int, image_shape: Tuple, bbox: List|str):
    """
    edge_thresh_x: both left and right
    edge_thresh_y: both top and bottom
    """
    if type(bbox) == str:
        numbers = bbox.strip('[]').split(',')
        prompt_coords_list = list(map(int, numbers))
        x_min, y_min, x_max, y_max = prompt_coords_list
    elif type(bbox) == list:
        x_min, y_min, x_max, y_max = bbox
    else:
        raise TypeError("bbox must be str or list.")

    image_shape_x = image_shape[1]
    image_shape_y = image_shape[0]

    if (x_min < edge_thresh_x) or (x_max > image_shape_x - edge_thresh_x):
        return True

    if (y_min < edge_thresh_y) or (y_max > image_shape_y - edge_thresh_y):
        return True

    return False

def get_mask(dir: str):
    patches_list = []
    for file in glob.glob(os.path.join(dir, '*.bmp')):
        file_name_bmp = file.replace('\\', '/').split('/')[-1]
        file_name = file_name_bmp.split('.')[0]
        posi, posj = file_name.split('_')[1:]
        mask = cv2.imread(os.path.join(dir, file_name_bmp))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        patches_list.append((mask_gray, posi, posj))

    return patches_list

def concat_mask(dir: str):  # operate on a list of (img_gray, posi, posj)
    """
    Reconstruct a full-size mask image by stitching together mask patches.

    This function assumes that a large image was previously divided into
    smaller overlapping or non-overlapping patches, and that each patch was
    processed separately to produce a binary/gray-scale mask. Each patch is
    stored as a .bmp file whose filename encodes its grid position in the
    form:

        <anything>_<rowIndex>_<colIndex>.bmp

    where:
        - rowIndex is the patch row index in the original grid
        - colIndex is the patch column index in the original grid

    The function:
        1. Loads all *.bmp mask patches in the directory.
        2. Parses the row/column indices from the filenames.
        3. Sorts patches by row, then column.
        4. Concatenates patches horizontally within each row.
        5. Concatenates all rows vertically.

    :param
    ----------
    dir : str
        Directory containing mask patches. All files must be .bmp images whose
        filenames include row and column indices as described above.

    :return
    -------
    full_image : np.ndarray
        The reconstructed full mask image formed by stitching the patches
        together. Pixel type and channel count match those of the patch images.

    Notes
    -----
    - All patches in a given row must have the same height.
    - All rows must have the same total width once concatenated.
    - No geometric blending or padding is applied: patches are joined directly
      edge-to-edge based on their grid indices.
    """

    patches = get_mask(dir)
    organized_patches = {}
    for patch_img, i, j in patches:
        i = int(i)
        j = int(j)
        if i not in organized_patches:
            organized_patches[i] = {}
        organized_patches[i][j] = patch_img

    concatenated_rows = []
    for i in sorted(organized_patches.keys()):
        row_patches = [organized_patches[i][j] for j in sorted(organized_patches[i].keys())]
        concatenated_row = cv2.hconcat(row_patches)  # Concatenate patches horizontally
        concatenated_rows.append(concatenated_row)

    full_mask = cv2.vconcat(concatenated_rows)

    return full_mask

def detect_droplet_regions(
    binary_mask: np.ndarray,
    min_area: int = 15,
    edge_thresh: int = 25
) -> List[Dict]:
    """
    Detect connected droplet regions from a binary mask.

    :param
    ----------
    binary_mask : np.ndarray
        2D binary mask (0/1).
    min_area : int
        Minimum pixel area to keep a region.

    :return
    -------
    droplets : list of dict
        Each dict contains:
            - bbox      : (min_row, min_col, max_row, max_col)
            - area      : int
            - centroid  : (row, col)
            - coords    : list of (row, col)

    mask_coords: dict with key being 'bbox' and val being 'coords'
    """
    labeled = label(binary_mask)
    regions = regionprops(labeled)

    droplets = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        if region.area < min_area:
            continue
        if filter_edge_bi_sides(edge_thresh, edge_thresh, binary_mask.shape, [min_col, min_row, max_col, max_row]):
            continue

        object_area = region.area
        object_perimeter = region.perimeter
        width = max_col - min_col
        height = max_row - min_row
        object_aspect_ratio = width / height if height != 0 else 0

        droplet = {
            "bbox": str([min_col, min_row, max_col, max_row]),
            "area": object_area,
            "perimeter": object_perimeter,
            "aspect_ratio": object_aspect_ratio,
            "centroid": region.centroid,         # (row, col)
            "coords": region.coords.tolist(),    # list of [row, col]
        }
        droplets.append(droplet)


    return droplets


