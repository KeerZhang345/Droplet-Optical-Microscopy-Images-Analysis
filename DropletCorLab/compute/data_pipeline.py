from DropletCorLab.common import load_pickle, save_csv
from typing import List, Any, Tuple, Dict
import numpy as np
from DropletCorLab.compute.pca import pca_given_n
import pandas as pd
import os
import cv2
from DropletCorLab.compute.pca import pca_visulization
from DropletCorLab.compute.clustering import perform_clustering, Single_datapoint_prob
import matplotlib.pyplot as plt
import ast
from DropletCorLab.preprocessing import image_preprocessing

def combine_raw_features_across_frames(raw_features_dir: str,
                             frame_IDs_all: List[int]
                             ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load relevant pre-computed raw features. Each file is identified as f'{frame_ID}_raw.pkl'

    :param
    ----------
    raw_features_dir : str
        dir storing raw features.
    frame_IDs_all : List[int]
        ID of frames to be analyzed.

    :return
    -------
    data_arg: Tuple, containing:
        - all_features_identifiers: dict
            each pair is global_enumerator: (frame_ID, bbox_string)
        - all_inner_color_features: np.ndarray
            all raw inner color features (n_bbox*n_frames, n_inner_color_features)
        - all_outer_color_features: np.ndarray
            all raw outer color features (n_bbox*n_frames, n_outer_color_features)
        - all_geo_features: np.ndarray
            all geo color features (n_bbox*n_frames, n_geo_features),
            in the order of 'area', 'perimeter', 'aspect_ratio'.

    """


    all_features_identifiers = {}
    all_inner_color_features = []
    all_outer_color_features = []
    all_geo_features = []

    current_index_offset = 0

    for frame_ID in frame_IDs_all:
        raw_features_path = os.path.join(raw_features_dir, f'{frame_ID}_raw.pkl')
        data = load_pickle(raw_features_path)

        features_identifiers = data["identifiers"]               # {local_idx: (frame_ID, bbox)}
        inner_color_features = data["inner_color_features"]      # ndarray (N, inner_dim)
        outer_color_features = data["outer_color_features"]      # ndarray (N, outer_dim)
        geo_features = data["geo_features"]                      # ndarray (N, 3) --> [area, perim, aspect]

        # Shift local droplet indices to global indexing
        updated_features_identifiers = {
            current_index_offset + idx: value
            for idx, value in features_identifiers.items()
        }

        # Save identifiers
        all_features_identifiers.update(updated_features_identifiers)

        # Collect features
        all_inner_color_features.append(inner_color_features)
        all_outer_color_features.append(outer_color_features)
        all_geo_features.append(geo_features)

        current_index_offset += len(features_identifiers)

    # Concatenate all frames
    all_inner_color_features = np.concatenate(all_inner_color_features, axis=0)
    all_outer_color_features = np.concatenate(all_outer_color_features, axis=0)
    all_geo_features = np.concatenate(all_geo_features, axis=0)  # shape (total_droplets, 3)

    data_args = (all_features_identifiers, all_inner_color_features, all_outer_color_features, all_geo_features)

    return data_args


def pca_visualization_on_combined_dataset(data_args, roi_key='', scaling_method: str='') -> None:
    """
    Visualize PCA result with cumulative explained variance and scree plot.

    :param
    ----------
    ori_data : Any
        array-like of shape (n_samples, n_features).
    roi_key: str
        which ROIs to analyze. Choose from 'inner' and 'outer'.
    scaling_method : str
        data scaling method before PCA. Choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.

    :return
    -------
    None.

    """

    all_features_identifiers, all_inner_color_features, all_outer_color_features, all_geo_features = data_args
    if roi_key == 'inner':
        pca_visulization(all_inner_color_features, scaling_method)
    elif roi_key == 'outer':
        pca_visulization(all_outer_color_features, scaling_method)
    else:
        raise ValueError("roi_key must be either 'inner' or 'outer'.")


def pca_on_combined_dataset(data_args: Tuple,
                            raw_features_dir: str,
                            num_pc_inner: int,
                            num_pc_outer: int,
                            scaling_method: str,
                            only_compute_weights: bool = False,
                            save_df: bool = False,
                            roi_frac: float = 0.1
                            ) -> Any:

    """
    Perform PCA on pre-loaded raw features.

    :param
    ----------
    data_args : Tuple
        direct output from combine_raw_features_across_frames()
    raw_features_dir: str,
        dir storing raw features.
    num_pc_inner: int,
        number of principal components retained for inner ROIs.
    num_pc_outer: int,
        number of principal components retained for outer ROIs.
    scaling_method : str
        data scaling method before PCA. Choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.
    only_compute_weights: bool
        if enabled, normalized pc weights (individual explained variance) is returned
        (inner_ROI_weights, outer_ROI_weights)
    save_df: bool = False
        whether saving the computed pc_df or not.
        if enabled, the computed df is saved to a folder named 'pc_df' (will be automatically created if non-exist)
        which is in parallel to the folder storing raw features, in the name of f'inner_{num_pc_inner}_outer_{num_pc_outer}.csv'
    roi_frac: float
        outer roi frac previously used to computed raw features.
        mainly for documentation purpose.

    :return
    -------
    returns (norm_inner_ROI_weights, norm_outer_ROI_weights) if only_compute_weights is enabled otherwise the (computed pc_df, (norm_inner_ROI_weights, norm_outer_ROI_weights)).

    """


    all_features_identifiers, all_inner_color_features, all_outer_color_features, all_geo_features = data_args

    # PCA reduction
    pca_df_inner, _, normalized_weights_inner, _ = pca_given_n(
        ori_data=all_inner_color_features,
        n=num_pc_inner,
        scaling_method=scaling_method
    )
    pca_df_outer, _, normalized_weights_outer, _ = pca_given_n(
        ori_data=all_outer_color_features,
        n=num_pc_outer,
        scaling_method=scaling_method
    )

    if only_compute_weights:
        return normalized_weights_inner, normalized_weights_outer

    # Rename PC columns
    pca_df_inner = pca_df_inner.rename(columns=lambda x: f"inner_{x}")
    pca_df_outer = pca_df_outer.rename(columns=lambda x: f"outer_{x}")

    # Combine PCA features
    combined_pca_df = pd.concat([pca_df_inner, pca_df_outer], axis=1)

    # Attach identifiers as index
    identifiers = [all_features_identifiers[idx] for idx in combined_pca_df.index]
    combined_pca_df.index = identifiers

    if isinstance(combined_pca_df.index[0], tuple):
        combined_pca_df[['frame_ID', 'bbox']] = pd.DataFrame(
            combined_pca_df.index.tolist(),
            index=combined_pca_df.index
        )

    combined_pca_df = combined_pca_df.reset_index()

    # Convert geometry into DataFrame (aligned automatically by index order)
    geo_df = pd.DataFrame(
        all_geo_features,
        columns=['area', 'perimeter', 'aspect_ratio']
    )

    # Merge everything
    result_df = combined_pca_df.merge(geo_df, left_index=True, right_index=True, how='inner')

    if save_df:
        df_dir = os.path.join(os.path.dirname(raw_features_dir), f'pc_df_roi_{roi_frac}')
        os.makedirs(df_dir, exist_ok=True)
        save_csv(result_df, os.path.join(df_dir, f'inner_{num_pc_inner}_outer_{num_pc_outer}.csv'))

    return result_df, (normalized_weights_inner, normalized_weights_outer)

def primary_clustering(pc_df: pd.DataFrame,
                       frame_IDs: List[int],
                       num_clusters: int,
                       scaling_method: str,
                       clustering_method: str,
                       roi_key: str,
                       show_center: bool = False,
                       trace_back: bool = True,
                       image_dir_: str = '',
                       pos_ref_ID: int = 0,
                       bri_ref_ID: int = 0,
                       mask_dict: Dict = None,
                       num_segs: int = 0
                       ) -> Tuple[np.ndarray, List, List[int]]:

    """
    Perform clustering on end stage frame(s) pc features

    :param
    ----------
    pc_df: pd.DataFrame
        pre-computed pc_df.
    frame_IDs: List[int]
        end stage frame IDs.
    num_clusters: int
        number of clusters.
    scaling_method: str
        data scaling method before clustering.
    clustering_method: str
        method used for clustering.
    roi_key: str
        which ROIs to analyze. Choose from 'inner' and 'outer'.
    show_center: bool
        whether to show cluster center or not.
    trace_back: bool
        whether to trace back to original segments

    (optional, only relevant when trace_back = True)
    image_dir_: str
        parent dir where original images are saved
    pos_ref_ID: int
        positional alignment reference image
    bri_ref_ID: int
        brightness alignment reference image
    mask_dict: Dict
        coords dict of each mask
    num_segs: int = 0
        number of example segments to show, if not positive integers, all segments will be shown.


    :return
    ----------
    counts: np.ndarray
        number of instances in each cluster.
    prob_args: contains following
        - trained_classifier: object
            classifier trained on end of stage pc features.
        - clustering_method: str
            passed clustering method.
        - prob_intermediate: np.ndarray
            computed cluster center.
        - pca_scaler: object
            scaler object fitted on end of stage pc features
        - scaling_method: str
            selected scaling method to fit pca_scaler.
    frame_IDs: List[int]
        end stage frame IDs.
    """

    for x in frame_IDs:
        if x not in pc_df['frame_ID']:
            raise ValueError("The frame_ID not exist in pc_df.")

    subset_all = pc_df[(pc_df['frame_ID'].isin(frame_IDs))]

    if roi_key == 'inner':
        pc_columns = list(subset_all.columns[subset_all.columns.str.startswith('inner_')])
    elif roi_key == 'outer':
        pc_columns = list(subset_all.columns[subset_all.columns.str.startswith('outer_')])
    else:
        raise ValueError("roi_key must be either 'inner' or 'outer'.")


    features_df, avg_score, prob_args, counts = perform_clustering(pc_df=subset_all,
                                                                   num_clusters=num_clusters,
                                                                   clustering_method=clustering_method,
                                                                   scaling_method=scaling_method,
                                                                   pc_columns=pc_columns,
                                                                   show_center=show_center
                                                                   )

    print(avg_score, prob_args)

    if trace_back:
        pos_ref = os.path.join(image_dir_, f"{pos_ref_ID}.jpg")
        bri_ref = os.path.join(image_dir_, f"{bri_ref_ID}.jpg")
        pos_ref_rgb = cv2.cvtColor(cv2.imread(pos_ref), cv2.COLOR_BGR2RGB)
        bri_ref_rgb = cv2.cvtColor(cv2.imread(bri_ref), cv2.COLOR_BGR2RGB)

        seg_list_all = []
        for frame_ID in frame_IDs:
            image_path = os.path.join(image_dir_, f"{frame_ID}.jpg")
            image_rgb_raw = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            seg_list = get_seg_list(image_rgb_raw, pos_ref_rgb, bri_ref_rgb, mask_dict, frame_ID)
            seg_list_all.extend(seg_list)

        trace_back_func(seg_list_all, features_df, num_clusters, num_segs)

    return counts, prob_args, frame_IDs


def get_seg_list(
        image_rgb_raw,
        image_pos_ref,
        image_bri_ref,
        mask_dict: Dict,
        frame_ID: int
) -> List:

    image_rgb = image_preprocessing(image_rgb_raw, image_pos_ref, image_bri_ref)

    seg_list =[]
    for bbox, coords in mask_dict.items():
        coords = np.array(coords)  # shape (N, 2) → rows, cols

        rows = coords[:, 0]
        cols = coords[:, 1]

        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        # crop bounding rectangle
        crop = image_rgb[rmin:rmax + 1, cmin:cmax + 1].copy()

        mask = np.zeros(crop.shape[:2], dtype=bool)
        mask[rows - rmin, cols - cmin] = True

        crop_masked = crop.copy()
        white = np.iinfo(crop.dtype).max if np.issubdtype(crop.dtype, np.integer) else 1.0
        crop_masked[~mask] = white

        identifier = (frame_ID, bbox)

        seg_list.append((identifier, crop_masked))

    return seg_list



def trace_back_func(seg_list, features_df, num_clusters, num_segs) -> None:

    # --- Map feature-row index → identifier tuple ---
    identifiers = [
        ast.literal_eval(item)
        for item in features_df['index'].tolist()
    ]
    features_identifiers = {
        idx: identifier
        for idx, identifier in enumerate(identifiers)
    }

    # --- Build mapping identifier → cropped image ---
    identifier_to_cropped_seg = {}
    for identifier, cropped_seg in seg_list:
        identifier_to_cropped_seg[identifier] = cropped_seg

    # --- Initialize output dicts ---
    seg_image = {f"seg_cluster_{k}": [] for k in range(num_clusters)}
    seg_ID =     {f"seg_cluster_{k}": [] for k in range(num_clusters)}

    # --- Assign each droplet to its cluster bucket ---
    for idx, cluster_label in enumerate(features_df['cluster']):
        identifier = features_identifiers[idx]
        cropped_seg = identifier_to_cropped_seg[identifier]

        seg_image[f"seg_cluster_{cluster_label}"].append(cropped_seg)
        seg_ID[f"seg_cluster_{cluster_label}"].append(identifier)

    # --- Optional visualization ---

    for cluster_idx in range(num_clusters):
        imgs = seg_image[f"seg_cluster_{cluster_idx}"]

        if len(imgs) == 0:
            continue

        if (type(num_segs) == int) and (num_segs > 0):
            select_idx = np.random.randint(0, len(imgs), min(num_segs, len(imgs)))
            imgs = [imgs[i] for i in select_idx]

        n = len(imgs)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(24, max(3, nrows*2)))

        axs = np.atleast_1d(axs).ravel()

        for ax, seg in zip(axs, imgs):
            ax.imshow(seg)
            ax.axis('off')

        for ax in axs[len(imgs):]:
            ax.axis('off')

        fig.suptitle(f"Cluster {cluster_idx}")
        plt.show()


def update_clustering(
        pc_df: pd.DataFrame,
        frame_IDs: List[int],
        frame_IDs_update: List[int],
        dro_idx: int,
        num_clusters: int,
        scaling_method: str,
        clustering_method: str,
        roi_key: str,
        show_center: bool = False,
        ) -> Tuple[np.ndarray, List]:
    """
    Update cluster centroid from end stage frame(s) pc features with early stage frame(s) pc features.

    :param
    ----------
    pc_df: pd.DataFrame
        pre-computed pc_df.
    frame_IDs: List[int]
        end stage frame IDs.
    frame_IDs_update: List[int]
        early stage frame IDs for centroid.
    dro_idx: int
        cluster index corresponding to non-corroding droplets.
    num_clusters: int
        number of clusters.
    scaling_method: str
        data scaling method before clustering.
    clustering_method: str
        method used for clustering.
    roi_key: str
        which ROIs to analyze. Choose from 'inner' and 'outer'.
    show_center: bool
        whether to show cluster center or not.

    :return
    ----------
    counts: np.ndarray
        number of instances in each cluster.
    prob_args: contains following
        - trained_classifier: object
            classifier trained on end stage pc features.
        - clustering_method: str
            passed clustering method.
        - prob_intermediate: np.ndarray
            computed cluster center, updated by initial frame(s) pc.
        - pca_scaler: object
            scaler object fitted on end of stage pc features (initial pc features are then transformed with this fitted scaler)
        - scaling_method: str
            selected scaling method to fit pca_scaler.
    """

    for x in frame_IDs + frame_IDs_update:
        if x not in pc_df['frame_ID']:
            raise ValueError("The frame_ID not exist in pc_df.")

    subset_all = pc_df[(pc_df['frame_ID'].isin(frame_IDs))]
    subset_all_update = pc_df[(pc_df['frame_ID'].isin(frame_IDs_update))]

    if roi_key == 'inner':
        pc_columns = list(subset_all.columns[subset_all.columns.str.startswith('inner_')])
    elif roi_key == 'outer':
        pc_columns = list(subset_all.columns[subset_all.columns.str.startswith('outer_')])
    else:
        raise ValueError("roi_key must be either 'inner' or 'outer'.")


    features_df, avg_score, prob_args, counts = perform_clustering(pc_df=subset_all,
                                                                   num_clusters=num_clusters,
                                                                   clustering_method=clustering_method,
                                                                   scaling_method=scaling_method,
                                                                   pc_columns=pc_columns,
                                                                   show_center=show_center,
                                                                   update_centroid_with_initial_frame=True,
                                                                   pc_df_initial=subset_all_update,
                                                                   dro_idx=dro_idx,
                                                                   merge_initial_cluster_color=True
                                                                   )

    print(avg_score, prob_args)

    return counts, prob_args


def merge_centers(cluster_index_list, centers_all, counts):
    """
    Compute the overall center of several clusters (weighted by sample number).
    """

    try:
        if len(cluster_index_list) == 1:
            return centers_all[cluster_index_list[0]]
        elif len(cluster_index_list) == 0:
            raise ValueError("Cluster index must not be empty.")
        else:
            weighted_center = np.zeros_like(centers_all[0])  # Same dimension as a single cluster center
            total_count = 0

            for i in cluster_index_list:  # Iterate over non-corroding cluster indices
                weighted_center += counts[i] * centers_all[i]
                total_count += counts[i]

            weighted_center /= total_count
            return weighted_center
    except IndexError:
        raise IndexError("Cluster index must not be non-negative integers.")


def refine_centers(prob_args: List,
                   counts: np.ndarray,
                   dro_idx_list: List[int],
                   corr_idx_list: List[int],
                   ) -> List:
    """
    Combine multiple cluster centers into 2.
    The returned cluster center always in the format of array([non-corroding center, corroding center]).

    :param
    ----------
    prob_args: List
        prob_args output from primary_clustering or update_clustering.
    counts: np.ndarray
        counts elements of each cluster, from primary_clustering or update_clustering.
    dro_idx_list:
        non-corroding clusters indexes.
    corr_idx_list:
        corroding clusters indexes.

    :return
    ----------
    prob_args: contains following
        - trained_classifier: object
            classifier trained on end stage pc features.
        - clustering_method: str
            passed clustering method.
        - prob_intermediate: np.ndarray
            computed cluster center, optionally updated by initial frame(s) pc.
            always two centers, in the format of array([non-corroding center, corroding center]).
        - pca_scaler: object
            scaler object fitted on end of stage pc features. Initial pc features are transformed with this fitted scaler if involved.
        - scaling_method: str
            selected scaling method to fit pca_scaler.
    """

    _, _, prob_intermediate, _, _ = prob_args
    corr_center = merge_centers(corr_idx_list, prob_intermediate, counts)
    no_corr_center = merge_centers(dro_idx_list, prob_intermediate, counts)

    refined_centers = [no_corr_center, corr_center]

    prob_args_copy = prob_args.copy()
    prob_args_copy[2] = refined_centers

    print(prob_args_copy)

    return prob_args_copy



def construct_p_worker(frame_ID: int,
                       pc_df: pd.DataFrame,
                       region: str,
                       prob_args_inner: List,
                       prob_args_outer: List,
                       pc_cols_inner: List,
                       pc_cols_outer: List,
                       norm_w_inner: np.ndarray,
                       norm_w_outer: np.ndarray,
                       distance_method: str = 'softmax'
                       ) -> Tuple[np.ndarray, Dict]:

    subset_all = pc_df[(pc_df['frame_ID'] == frame_ID)]

    if region == 'inner':
        features_inner = subset_all[pc_cols_inner].values
        features_outer = None
    elif region == 'outer':
        features_inner = None
        features_outer = subset_all[pc_cols_outer].values
    else:
        features_inner = subset_all[pc_cols_inner].values
        features_outer = subset_all[pc_cols_outer].values

    indexes = subset_all['index'].values

    probs_all = []
    probs_dict = {}
    for i in range(len(features_inner)):
        if region == 'inner':
            probs_inner = Single_datapoint_prob.predict_prob(features_inner[i], norm_w_inner,
                                                             *prob_args_inner, distance_method=distance_method)
            p_x = probs_inner[1]
            probs_ = [p_x]
        elif region == 'outer':
            probs_outer = Single_datapoint_prob.predict_prob(features_outer[i], norm_w_outer,
                                                             *prob_args_outer, distance_method=distance_method)
            p_y = probs_outer[1]
            probs_ = [p_y]
        else:
            probs_inner = Single_datapoint_prob.predict_prob(features_inner[i], norm_w_inner,
                                                                  *prob_args_inner, distance_method=distance_method)

            probs_outer = Single_datapoint_prob.predict_prob(features_outer[i], norm_w_outer,
                                                                  *prob_args_outer, distance_method=distance_method)

            p_x = probs_inner[1]
            p_y = probs_outer[1]

            probs_ = [p_x, p_y]

        probs_all.append(probs_)
        probs_dict[indexes[i]] = probs_

    probs_array = np.array(probs_all)

    return probs_array, probs_dict


def construct_p_all_frames(pc_df: pd.DataFrame,
                           region: str = '',
                           prob_args_inner: List = None,
                           prob_args_outer: List = None,
                           norm_w_inner: np.ndarray = None,
                           norm_w_outer: np.ndarray = None,
                           distance_method: str = 'softmax',
                           if_save: bool = False,
                           raw_features_dir: str = '',
                           roi_frac: float = 0.1
                           ) -> None:

    """
    Compute probability descriptors for all droplets in all frames.

    :param
    ----------
    pc_df: pd.DataFrame
        pre-computed pc_df.
    region: str
        which ROI to compute. Choose from 'inner', 'outer', or 'inner_outer'
    prob_args_inner: List
        prob_args output from primary_clustering, update_clustering or refine_centers for inner ROIs.
    prob_args_outer: List
        prob_args output from primary_clustering, update_clustering or refine_centers for outer ROIs.
    norm_w_inner: np.ndarray
        normalized weight from PCA for inner ROIs.
    norm_w_outer: np.ndarray
        normalized weight from PCA for outer ROIs.
    distance_method: str
        method for transforming distances between data point and cluster centers into probability indicators.
        choose from 'softmax' or 'inverse'
        - softmax
        p = np.exp(-distances) / np.sum(np.exp(-distances))
        - inverse
        p = (1 / distances) / np.sum(1 / distances)
    if_save: bool
        whether to save the computed p file

    (optional, only relevant when if_save = True)
    raw_features_dir: str
        dir storing raw features.
    roi_frac: float
        outer roi frac previously used to computed raw features.
        mainly for documentation purpose.

    :return
    ----------
    None
    """


    if region not in ['inner', 'outer', 'inner_outer']:
        raise ValueError("Region must be either 'inner', 'outer', or 'inner_outer'.")

    if region in ['inner', 'inner_outer']:
        if prob_args_inner is None:
            raise ValueError("prob_args_inner must not be None when region is 'inner' or 'inner_outer'.")
    elif region in ['outer', 'inner_outer']:
        if prob_args_outer is None:
            raise ValueError("prob_args_outer must not be None when region is 'outer' or 'inner_outer'.")

    if (norm_w_inner is None) or (norm_w_outer is None):
            raise ValueError("norm_w_inner and/or norm_w_outer not provided.")

    pc_cols_inner = list(pc_df.columns[pc_df.columns.str.startswith('inner_')])
    pc_cols_outer = list(pc_df.columns[pc_df.columns.str.startswith('outer_')])
    frame_IDs_all_ = pc_df['frame_ID'].unique()

    probs_dict_all = {}
    for frame_ID in frame_IDs_all_:
        probs, probs_dict = construct_p_worker(frame_ID,
                                               pc_df,
                                               region,
                                               prob_args_inner,
                                               prob_args_outer,
                                               pc_cols_inner,
                                               pc_cols_outer,
                                               norm_w_inner,
                                               norm_w_outer,
                                               distance_method)

        probs_dict_all = {**probs_dict_all, **probs_dict}

    if region == 'inner':
        columns = ['p_x']
    elif region == 'outer':
        columns = ['p_y']
    else:
        columns = ['p_x', 'p_y']

    dict_df = pd.DataFrame.from_dict(probs_dict_all, orient='index', columns=columns)
    dict_df.reset_index(inplace=True)
    dict_df.rename(columns={'index': 'index'}, inplace=True)
    result_df = pd.merge(pc_df, dict_df, on='index', how='left')

    if if_save:
        df_dir = os.path.join(os.path.dirname(raw_features_dir), f'pc_df_roi_{roi_frac}')
        save_csv(result_df, os.path.join(df_dir, f'inner_{len(pc_cols_inner)}_outer_{len(pc_cols_outer)}_pro_included.csv'))


