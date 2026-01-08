import numpy as np
import cv2
from typing import Tuple, List, Dict


def equivalent_diameter_from_area(area_pixels: int) -> float:
    """
    Compute equivalent diameter of a region from its area.

    Parameters
    ----------
    area_pixels : int
        Area in pixels.

    Returns
    -------
    eq_diameter : float
    """
    return 2.0 * np.sqrt(area_pixels / np.pi)


def build_binary_mask_from_coords(
    coords: List[List[int]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Build a binary mask from pixel coordinates.

    Parameters
    ----------
    coords : list of [row, col]
    image_shape : (H, W)

    Returns
    -------
    mask : np.ndarray
        Binary mask.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    coords_arr = np.asarray(coords)
    mask[coords_arr[:, 0], coords_arr[:, 1]] = 1
    return mask


def compute_inner_outer_roi(
    droplet: Dict,
    image_shape: Tuple[int, int],
    roi_outer_frac: float,
    binary_mask: np.ndarray
) -> Dict:

    coords_inner = droplet["coords"]
    area = droplet["area"]

    eq_dia = equivalent_diameter_from_area(area)
    dilation_radius = int(eq_dia * roi_outer_frac)

    if dilation_radius < 1:
        raise ValueError("Outer ROI radius too small; increase roi_outer_frac.")

    # Inner mask
    inner_mask = build_binary_mask_from_coords(coords_inner, image_shape)

    # Dilated mask → inner + outer
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    )

    dilated_mask = cv2.dilate(inner_mask, kernel, iterations=1)

    # Outer ring (exclude inner)
    outer_mask = (dilated_mask - inner_mask) > 0
    outer_mask = outer_mask.astype(np.uint8)

    # Exclude pixels belonging to other droplets
    outer_mask = outer_mask & (~binary_mask.astype(bool))
    outer_mask = outer_mask.astype(np.uint8)

    coords_outer = np.argwhere(outer_mask == 1).tolist()

    data = {
        "bbox": droplet["bbox"],
        "inner_mask": inner_mask,
        "outer_mask": outer_mask,
        "eq_diameter": eq_dia,
        "area": area,
        "aspect_ratio": droplet["aspect_ratio"],
        "perimeter": droplet["perimeter"],
        "coords_inner": coords_inner,
        "coords_outer": coords_outer,
    }

    return data

def compute_inner_outer_roi_all_droplets(
    droplets: List[Dict],
    image_shape: Tuple[int, int],
    roi_outer_frac: float,
    binary_mask: np.ndarray
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute inner and outer ROI masks for all droplets.

    :param
    ----------
    droplets: List[dict]
        Each dict contains:
            - "bbox": str, xyxy,
            - "area": float
            - "perimeter": float
            - "aspect_ratio": float
            - "centroid": tuple
            - "coords": List[List[int]]
    image_shape : (H, W)
    roi_outer_frac : float
        Outer ROI radius as fraction of equivalent diameter.
    binary_mask : np.ndarray
        Binary mask of all droplets (used to exclude neighbors).

    :return
    -------
    mask_coords_inner: dict
        each pair contain a key being bbox (xyxy) and val being list of coords ([row, col]) in inner ROI
    mask_coords_outer: dict
        each pair contain a key being bbox (xyxy) and val being list of coords ([row, col]) in outer ROI
    """

    masks_coords_inner = {}
    masks_coords_outer = {}
    geo = {}

    for droplet in droplets:
        data_i = compute_inner_outer_roi(droplet, image_shape, roi_outer_frac, binary_mask)
        masks_coords_inner[data_i["bbox"]] = data_i["coords_inner"]
        masks_coords_outer[data_i["bbox"]] = data_i["coords_outer"]
        geo[data_i["bbox"]] = {"area": data_i["area"], "perimeter": data_i["perimeter"], "aspect_ratio": data_i["aspect_ratio"]}

    return masks_coords_inner, masks_coords_outer, geo

