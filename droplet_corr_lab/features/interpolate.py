import numpy as np
import cv2


def build_cropped_interpolated_roi(
    image_rgb,
    coords,
    bbox,
    target_size,
):
    """
    image_rgb: aligned frame image (H, W, 3)
    coords: original-scale coords [[r,c], ...]
    bbox: [xmin, ymin, xmax, ymax] (original scale)
    """

    # build binary mask for this specific droplet, the whole canvas at the size if img_rgb
    mask_full = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    coords_arr = np.asarray(coords)
    mask_full[coords_arr[:, 0], coords_arr[:, 1]] = 1

    # --- crop ---
    xmin, ymin, xmax, ymax = eval(bbox)
    cropped_img = image_rgb[ymin:ymax, xmin:xmax]
    cropped_mask = mask_full[ymin:ymax, xmin:xmax]

    if cropped_img.size == 0:
        raise ValueError("Empty cropped ROI")

    # --- interpolate ---
    interp_img = cv2.resize(
        cropped_img,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_CUBIC
    )

    interp_mask = cv2.resize(
        cropped_mask,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    return interp_img, interp_mask

