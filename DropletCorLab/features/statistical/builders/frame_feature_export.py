from typing import Dict, List, Callable, Tuple, Optional
import os
import numpy as np
import gc
from DropletCorLab.features.statistical.builders.group_features import extract_features_for_groups
from DropletCorLab.features.statistical.sampling import grid_radial_slices, grid_ring_slices, grid_blocks, build_cropped_interpolated_roi
from DropletCorLab.preprocessing.aligment import estimate_translation
from DropletCorLab import save_pickle
from DropletCorLab.preprocessing import image_preprocessing
from DropletCorLab.features.statistical.descriptors import rgb_features, hsv_features, gradient_features, rgb_features_only_mean, hsv_features_only_mean
from multiprocessing import Pool


def descriptors_dict(des_key: List[str]) -> List[Callable]:
    des_dict = {
    'rgb_m_s': rgb_features,
    'hsv_m_s': hsv_features,
    'gradient': gradient_features,
    'rgb_m': rgb_features_only_mean,
    'hsv_m': hsv_features_only_mean,
    }

    return [des_dict[i] for i in des_key]

def filter_edge_one_side(edge_thresh_x: int, edge_thresh_y: int, image_shape: Tuple, bbox: List|str):
    """
    edge_thresh_x:
       > 0 --> shift to right
    edge_thresh_y:
       > 0 --> shift to bottom
    """
    if type(bbox) == str:
        numbers = bbox.strip('[]').split(',')
        prompt_coords_list = list(map(int, numbers))
        x_min, y_min, x_max, y_max = prompt_coords_list
    elif type(bbox) == list:
        x_min, y_min, x_max, y_max = bbox
    else:
        raise TypeError("bbox must be str or list.")

    if edge_thresh_x > 0:
        if x_min < edge_thresh_x:
            return True
    else:
        if x_max > image_shape[1] + edge_thresh_x:
            return True

    if edge_thresh_y > 0:
        if y_min < edge_thresh_y:
            return True
    else:
        if y_max > image_shape[0] + edge_thresh_y:
            return True

    return False


def build_per_droplet_features(
    image_rgb: np.ndarray,
    bbox,
    geo: Dict,
    coords_inner: List[List[int]],
    coords_outer: List[List[int]],
    frame_idx: int,
    method: str,
    param_inner,
    param_outer,
    target_size: Tuple[int, int],
    inner_descriptors: List[Callable],
    outer_descriptors: List[Callable],
) -> Dict[str, float]:

    features = {
        "frame_ID": frame_idx,
        "bbox": bbox,
        "area": geo["area"],
        "perimeter": geo["perimeter"],
        "aspect_ratio": geo["aspect_ratio"],
    }

    # Inner ROI
    img_i, mask_i = build_cropped_interpolated_roi(
        image_rgb, coords_inner, bbox, target_size
    )

    if method == "ring":
        groups_i = grid_ring_slices(mask_i, param_inner)
    elif method == "radial":
        groups_i = grid_radial_slices(mask_i, param_inner)
    else:
        bs, st = param_inner
        groups_i = grid_blocks(mask_i, bs, st)

    inner_feats = extract_features_for_groups(
        img_i, groups_i, inner_descriptors, "inner"
    )
    for f in inner_feats:
        features.update(f)

    # Outer ROI
    img_o, mask_o = build_cropped_interpolated_roi(
        image_rgb, coords_outer, bbox, target_size
    )

    if method == "ring":
        groups_o = grid_ring_slices(mask_o, param_outer)
    elif method == "radial":
        groups_o = grid_radial_slices(mask_o, param_outer)
    else:
        bs, st = param_outer
        groups_o = grid_blocks(mask_o, bs, st)

    outer_feats = extract_features_for_groups(
        img_o, groups_o, outer_descriptors, "outer"
    )
    for f in outer_feats:
        features.update(f)

    return features


def build_per_frame_features(args):
    image_rgb_raw, frame_idx = args

    global _GLOBAL_MASK_INNER, _GLOBAL_MASK_OUTER, _GLOBAL_GEOS
    global _GLOBAL_POS_REF, _GLOBAL_BRI_REF
    global _GLOBAL_INNER_DES, _GLOBAL_OUTER_DES
    global _GLOBAL_METHOD, _GLOBAL_PARAM_INNER, _GLOBAL_PARAM_OUTER
    global _GLOBAL_TARGET_SIZE, _GLOBAL_EDGE_THRESH, _GLOBAL_SAVE,_GLOBAL_SAVE_DIR

    edge_x, edge_y = estimate_translation(_GLOBAL_POS_REF, image_rgb_raw)
    edge_x = min(edge_x, -_GLOBAL_EDGE_THRESH) if edge_x < 0 else max(edge_x, _GLOBAL_EDGE_THRESH)
    edge_y = min(edge_y, -_GLOBAL_EDGE_THRESH) if edge_y < 0 else max(edge_y, _GLOBAL_EDGE_THRESH)

    image_rgb = image_preprocessing(image_rgb_raw, _GLOBAL_POS_REF, _GLOBAL_BRI_REF)
    image_shape = image_rgb.shape

    all_features = []


    for bbox in _GLOBAL_MASK_INNER.keys():

        if filter_edge_one_side(edge_x, edge_y, image_shape, bbox):
            continue

        feats = build_per_droplet_features(
            image_rgb=image_rgb,
            bbox=bbox,
            geo=_GLOBAL_GEOS[bbox],
            coords_inner=_GLOBAL_MASK_INNER[bbox],
            coords_outer=_GLOBAL_MASK_OUTER[bbox],
            frame_idx=frame_idx,
            method=_GLOBAL_METHOD,
            param_inner=_GLOBAL_PARAM_INNER,
            param_outer=_GLOBAL_PARAM_OUTER,
            target_size=_GLOBAL_TARGET_SIZE,
            inner_descriptors=_GLOBAL_INNER_DES,
            outer_descriptors=_GLOBAL_OUTER_DES,
        )

        all_features.append(feats)


    identifiers = {}
    inner_mat = []
    outer_mat = []
    geo_mat = []

    for i, f in enumerate(all_features):
        identifiers[i] = (f["frame_ID"], f["bbox"])

        inner_mat.append([v for k, v in f.items() if k.startswith("inner_")])
        outer_mat.append([v for k, v in f.items() if k.startswith("outer_")])
        geo_mat.append([f["area"], f["perimeter"], f["aspect_ratio"]])

    out = {
        "identifiers": identifiers,
        "inner_color_features": np.asarray(inner_mat, dtype=np.float32),
        "outer_color_features": np.asarray(outer_mat, dtype=np.float32),
        "geo_features": np.asarray(geo_mat, dtype=np.float32),
    }

    if _GLOBAL_SAVE:
        save_pickle(out, os.path.join(_GLOBAL_SAVE_DIR, f"{frame_idx}_raw.pkl"))
    else:
        return out

    del image_rgb_raw, image_rgb, all_features
    del inner_mat, outer_mat, geo_mat
    gc.collect()


def _init_worker(mask_inner,
                mask_outer,
                geos,
                pos_ref,
                bri_ref,
                inner_des,
                outer_des,
                method,
                param_inner,
                param_outer,
                target_size,
                edge_thresh,
                save_,
                save_dir
                 ):
    global _GLOBAL_MASK_INNER, _GLOBAL_MASK_OUTER, _GLOBAL_GEOS
    global _GLOBAL_POS_REF, _GLOBAL_BRI_REF
    global _GLOBAL_INNER_DES, _GLOBAL_OUTER_DES
    global _GLOBAL_METHOD, _GLOBAL_PARAM_INNER, _GLOBAL_PARAM_OUTER
    global _GLOBAL_TARGET_SIZE, _GLOBAL_EDGE_THRESH, _GLOBAL_SAVE, _GLOBAL_SAVE_DIR

    _GLOBAL_MASK_INNER = mask_inner
    _GLOBAL_MASK_OUTER = mask_outer
    _GLOBAL_GEOS = geos
    _GLOBAL_POS_REF = pos_ref
    _GLOBAL_BRI_REF = bri_ref
    _GLOBAL_INNER_DES = descriptors_dict(inner_des)
    _GLOBAL_OUTER_DES = descriptors_dict(outer_des)
    _GLOBAL_METHOD = method
    _GLOBAL_PARAM_INNER = param_inner
    _GLOBAL_PARAM_OUTER = param_outer
    _GLOBAL_TARGET_SIZE = target_size
    _GLOBAL_EDGE_THRESH = edge_thresh
    _GLOBAL_SAVE = save_
    _GLOBAL_SAVE_DIR = save_dir


def build_features(
    image_rgb_raw_all,
    image_pos_ref,
    image_bri_ref,
    frame_idx_all: List,
    mask_coords_inner: Dict,
    mask_coords_outer: Dict,
    method: str,
    inner_descriptors: List[str],
    outer_descriptors: List[str],
    geos_all: Dict,
    target_size: Tuple[int, int] = (32, 32),
    param_inner: int = 5,
    param_outer: int = 2,
    edge_thresh_mask: int = 25,
    save_features=True,
    saving_par_dir: str = '',
    n_workers: int = 4
) -> None:
    """
        Compute features (color+geo) for all droplets in all given frame using the same pre-computed binary mask.

        :param
        ----------
        image_rgb_raw_all: np.ndarray
            all raw rgb image before positional shift and brightness adjustment.
        image_pos_ref: np.ndarray
            rgb image for positional shift as reference.
        image_bri_ref: np.ndarray
            rgb image for brightness adjustment as reference.
        frame_idx_all: int
            all frame ID, or unique identifier of all raw rgb image
        mask_coords_inner: Dict
            coords of all droplets' inner ROI from the pre-computed binary mask.
        mask_coords_outer: Dict
            coords of all droplets' outer ROI from the pre-computed binary mask.
        method: str
            method from grouping, choose from
                - 'ring': slicing along the diameter to create non-overlap rings with equal width (outer_dia - inner_dia)
                    :param
                    param_inner or param_outer: int
                        Number of concentric rings.
                - 'radial': slicing into non-overlapping angular (radial) slices with equal angles
                    :param
                    param_inner or param_outer: int
                        Number of angular bins over [0, 2π).
                - 'blocks': slicing into partially overlapping spatial blocks
                    :param
                    param_inner or param_outer: Tuple[int, int]
                        [0]: Size of each square block.
                        [1]: Step size between block origins.
        inner_descriptors: List[str]
            choose from 'rgb_m_s', 'rgb_m', 'hsv_m_s', 'hsv_m', 'gradient'
        outer_descriptors: List[str]
            choose from 'rgb_m_s', 'rgb_m', 'hsv_m_s', 'hsv_m', 'gradient'
        geos_all: Dict,
            each droplet inner ROI's geometrical descriptors, including 'area', 'perimeter', 'aspect_ratio'
        target_size: Tuple[int, int]
            Target size for interpolation on each segment and mask, default = (32, 32)
            interpolation method:
            - image: cv2.INTER_CUBIC
            - mask: cv2.INTER_NEAREST
        param_inner: int = 5
            see 'method'.
        param_outer: int = 2,
            see 'method'.
        edge_thresh_mask: int = 25,
            edge droplet filtering when bbox falls in threshold region
        save_features=True,
            save computed dict file as f"{frame_idx}_raw.pkl"
            each dict contains:
                - identifiers: Dict[int: Tuple[frame_idx: int, bbox: str]],
                - inner_color_features: np.ndarray, shape = (num_droplets, num_inner_features),
                - outer_color_features: np.ndarray, shape = (num_droplets, num_outer_features),
                - geo_features: np.ndarray, shape = (num_droplets, num_geo_features=3),
        saving_par_dir: str = '',
            dir for saving computed raw features
        n_workers: int = 4
            number of worker for parallel processing. Default = 4. Set to 1 to disable parallel processing.

        :return
        -------
        None
    """

    worker_args = [
        (
            image_rgb_raw,
            frame_idx
        )
        for image_rgb_raw, frame_idx in zip(image_rgb_raw_all, frame_idx_all)
    ]

    if n_workers > 1:
        with Pool(
                n_workers,
                initializer=_init_worker,
                initargs=(
                        mask_coords_inner,
                        mask_coords_outer,
                        geos_all,
                        image_pos_ref,
                        image_bri_ref,
                        inner_descriptors,
                        outer_descriptors,
                        method,
                        param_inner,
                        param_outer,
                        target_size,
                        edge_thresh_mask,
                        save_features,
                        saving_par_dir,
                ),
        ) as pool:
            pool.map(build_per_frame_features, worker_args, chunksize=1)

    else:
        _init_worker(
            mask_coords_inner,
            mask_coords_outer,
            geos_all,
            image_pos_ref,
            image_bri_ref,
            inner_descriptors,
            outer_descriptors,
            method,
            param_inner,
            param_outer,
            target_size,
            edge_thresh_mask,
            save_features,
            saving_par_dir,
        )
        for a in worker_args:
            build_per_frame_features(a)



