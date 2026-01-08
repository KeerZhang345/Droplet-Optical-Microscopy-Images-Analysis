import os
import glob
import cv2
import numpy as np
from typing import List


def crop_image_to_patches(
    image_ori: np.ndarray,
    mask_ID: int,
    num_patches_width: int = 2,
    num_patches_height: int = 2,
    crop_path: str = "",
    save_patches: bool = False
) -> List[np.ndarray]:
    """
    Split an image into a grid of patches.

    Rule:
        • All interior patches are equal size
        • Last row/column absorb remaining pixels (no loss)

    :param
    ----------
    image_ori : np.ndarray (H,W,3)
        Input image (RGB or BGR).

    mask_ID : int or str
        ID used in patch filenames.

    num_patches_width : int
        Number of patches across the width.

    num_patches_height : int
        Number of patches across the height.

    crop_path : str
        Directory to save patches (if enabled).

    save_patches : bool
        If True, clear crop_path before saving, and write patches as:
            {mask_ID}_{row}_{col}.bmp

    :return
    -------
     patches:
            [
              ((row_idx, col_idx), patch_rgb),
              ...
            ]
    """

    H, W = image_ori.shape[:2]
    patch_w = W // num_patches_width
    patch_h = H // num_patches_height

    patches_all = []

    for i in range(num_patches_height):
        for j in range(num_patches_width):

            x_start = j * patch_w
            y_start = i * patch_h

            # last column / row take the remainder
            x_end = W if j == num_patches_width - 1 else x_start + patch_w
            y_end = H if i == num_patches_height - 1 else y_start + patch_h

            patch = image_ori[y_start:y_end, x_start:x_end]
            patches_all.append(((i, j), patch))

    # ---------- optional saving ----------
    if save_patches and crop_path:

        os.makedirs(crop_path, exist_ok=True)

        for f in glob.glob(os.path.join(crop_path, "*")):
            os.remove(f)

        for (i, j), patch_rgb in patches_all:
            filename = f"{mask_ID}_{i}_{j}.bmp"
            filepath = os.path.join(crop_path, filename)

            # convert back to BGR for cv2.imwrite
            patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, patch_bgr)

    return patches_all
