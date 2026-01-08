from droplet_corr_lab.common import load_csv
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, lognorm, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
from matplotlib.cm import ScalarMappable
from droplet_corr_lab.preprocessing.aligment import image_preprocessing
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
import math

def classify_droplet_spatial_clustering(mask_ID: int,
                                p_df: pd.DataFrame,
                                thresh: float = None) -> pd.DataFrame:
    """
    Analyze whether a droplet is spatially isolated, or appears as in a cluster.

    :param
    ----------
    mask_ID: int
        frame_ID of the image to generate the binary mask
    p_df: pd.DataFrame
        pre-computed df containing probability descriptors for all droplets in all frames
    thresh: float
        threshold of determining whether a droplet is considered as 'in a cluster' or 'isolated'.
        default = None
        if None is passed, thresh is automatically calculated from the median of each droplets' nearest neighbor distance.
        otherwise use the passed value.

    :return:
    p_df: pd.DataFrame
        updated df containing the clustering info.
    """

    unique_droplets = p_df[p_df['frame_ID'] == mask_ID][['bbox', 'x_center', 'y_center', 'diameter']].reset_index(
        drop=True)

    if len(unique_droplets) <= 1:
        p_df['is_clustered'] = False
        return p_df

    unique_droplets['radius'] = unique_droplets['diameter'] / 2

    locations = unique_droplets[['x_center', 'y_center']].values
    center_dist_matrix = squareform(pdist(locations))

    radii = unique_droplets['radius'].values
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    edge_dist_matrix = center_dist_matrix - radii_sum_matrix

    np.fill_diagonal(edge_dist_matrix, np.inf)
    nearest_neighbor_dist = np.maximum(0, edge_dist_matrix.min(axis=1))
    unique_droplets['nearest_neighbor_dist'] = nearest_neighbor_dist

    thresh = unique_droplets['nearest_neighbor_dist'].median() if thresh is None else thresh
    print(f"Thresh used: {thresh}")

    unique_droplets['is_clustered'] = unique_droplets['nearest_neighbor_dist'] <= thresh

    p_df = pd.merge(p_df, unique_droplets[['bbox', 'is_clustered']],
                    on=['bbox'], how='left')
    return p_df

def get_center(p_df):
    coords = p_df['bbox'].str.strip('[]').str.split(',', expand=True).astype(int)

    x_center = (coords[0] +coords[2])//2
    y_center = (coords[1] +coords[3])//2

    p_df['x_center'] = x_center
    p_df['y_center'] = y_center

    return p_df


def find_flat_regions(
        y,
        slope_threshold=0.005,
        min_length=5,
        min_value=0.5):

    dy = np.gradient(y)

    flat = np.abs(dy) < slope_threshold
    starts, ends = [], []

    i = 0
    while i < len(flat):
        if flat[i]:
            j = i
            while j < len(flat) and flat[j]:
                j += 1

            if (j - i) >= min_length and np.mean(y[i:j]) >= min_value:
                starts.append(i)
                ends.append(j)

            i = j
        else:
            i += 1

    return list(zip(starts, ends))


def find_end_plateau(
        y,
        search_start=None,
        slope_threshold=0.005,
        min_length=5,
        min_value=0.5):

    if search_start is None:
        search_start = len(y) // 2   # usually appears after halfway

    flat_regions = find_flat_regions(
        y,
        slope_threshold=slope_threshold,
        min_length=min_length,
        min_value=min_value
    )

    # keep only regions starting after search_start
    late_regions = [r for r in flat_regions if r[0] >= search_start]

    if not late_regions:
        return None

    # pick the earliest of the late plateaus
    return late_regions[0]

def find_initial_plateau(
        y,
        end_plateau,
        slope_threshold=0.005,
        min_length=5,
        ini_end_diff=0.2):

    if end_plateau is None:
        return None

    end_start, end_end = end_plateau # index
    end_mean = np.mean(y[end_start:end_end]) # value

    flat_regions = find_flat_regions(
        y,
        slope_threshold=slope_threshold,
        min_length=min_length,
        min_value=0.0  # allow low plateau
    )

    candidates = [r for r in flat_regions if r[1] < end_start]

    if not candidates:
        return None

    # choose the highest-mean plateau that is still < final plateau - ini_end_diff
    candidates = [
        r for r in candidates if np.mean(y[r[0]:r[1]]) < (end_mean - ini_end_diff)
    ]

    if not candidates:
        return None

    # pick last one (closest to rise onset)
    return candidates[-1]

def get_rise_start_index(initial_plateau):
    if initial_plateau is None:
        return 2   # scientifically-motivated fallback
    else:
        return initial_plateau[1]


def get_p(p_df: pd.DataFrame,
          bbox: str,
          region: str = ''):

    if region == 'inner':
        key = 'p_x'
    elif region == 'outer':
        key = 'p_y'
    else:
        raise ValueError("region must be 'inner' or 'outer'.")
    ps = p_df[p_df['bbox'] == bbox][key].values
    return ps

def get_curve_statistics(
        y: np.ndarray,
        w_sg: int,
        poly: int,
        slope_threshold: float,
        min_length: int,
        min_value: float,
        search_start: int,
        T: float) -> Tuple[Dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    """
    Analyse a normalized temporal curve (0–1) to detect end-stage plateau behaviour
    and extract key kinetic indicators.

    The curve is first smoothed using a Savitzky–Golay filter. A late-stage plateau
    is detected as a contiguous region where the absolute slope remains below a
    threshold for at least `min_length` samples, and whose mean value exceeds
    `min_value`. The search for the end plateau begins at `search_start`
    (default = halfway point of the signal).

    If an end-stage plateau exists, the function also attempts to detect an
    early-stage (baseline) plateau that occurs before the rise and whose mean
    remains below the end-plateau level. If no such baseline plateau exists,
    the rise is assumed to begin near the start of the experiment
    (index = 2 by convention).

    Transition onset, plateau onset, midpoint timing, slopes and AUC are then
    computed from the smoothed curve.

    :param
    ----------
    y : np.ndarray
        1-D array of normalized signal values in the range [0, 1].
    w_sg : int, optional
        Window length for Savitzky–Golay smoothing (default = 23).
        Must be an odd integer ≥ polynomial order + 2.
    poly : int, optional
        Polynomial order for Savitzky–Golay smoothing (default = 2).
    slope_threshold : float, optional
        Maximum absolute gradient considered "flat" when detecting plateaus
        (default = 0.005).
    min_length : int, optional
        Minimum number of consecutive flat points required to define a plateau
        (default = 5).
    min_value : float, optional
        Minimum mean value of the late-stage plateau. This value is
        scientifically meaningful for your system and defaults to 0.5.
    search_start : int or None, optional
        Index from which to begin searching for an end-stage plateau.
        If None, defaults to half the curve length.
    T : float, optional
        Time resolution scaling factor. The time axis is defined as
        ``np.linspace(0, T * len(y), len(y))`` (default = 2.0 minutes per sample).

    :return
    -------
    stats : dict
        Dictionary containing:
            - 'Plateau Exists' : int (1 if a late-stage plateau is detected, else 0)
            - 'Midpoint' : float
                Time corresponding to the midpoint between transition onset and
                plateau onset (or time of maximum if no plateau exists).
            - 'Area Under Curve (AUC)' : float
                Integral of the smoothed signal.
            - 'Average Local Slope' : float or None
                Mean of local slopes within the transition region.
            - 'Transition Overall Slope' : float or None
                Global slope between transition onset and plateau onset.
            - 'Transition Time' : float or None
                Time difference between plateau onset and transition onset.
            - 'Plateau Value' : float or None
                Mean value of the end-stage plateau. None if no plateau exists.
            - 'End Plateau' : tuple or None
                (start_index, end_index) of the detected end plateau interval.
                None if no plateau exists.
            - 'Rise Start Idx' : int or None
                Index marking the end of the baseline plateau (or 2 if no baseline
                plateau exists). None if no plateau is detected at all.

    data : tuple
        A tuple ``(x, y, y_s)`` containing:
            - x: ndarray
                Time axis corresponding to the signal.
            - y: ndarray
                Original input signal
            - y_s: ndarray
                Smoothed version of the input signal.
    notes
    -----
    - All calculations are performed on the smoothed curve.
    - If no late-stage plateau is detected, transition-related quantities are None
      and the midpoint corresponds to the time of maximum signal.
    - If a baseline plateau cannot be detected, the rise onset is assumed to occur
      near the beginning (index = 2), consistent with the physical model.

    """


    # smooth
    if w_sg >= len(y):
        w_sg = len(y) - (1 - len(y) % 2)  # largest odd

    y_s = savgol_filter(y, window_length=w_sg, polyorder=poly, mode='nearest')
    x = np.linspace(0, T * len(y_s), len(y_s))  # time (min)

    # --- detect final plateau ---
    end_plateau = find_end_plateau(
        y_s,
        search_start=search_start,
        slope_threshold=slope_threshold,
        min_length=min_length,
        min_value=min_value
    )

    if end_plateau:
        plateau_exists = 1

        # --- detect optional early plateau ---
        init_plateau = find_initial_plateau(
            y_s,
            end_plateau=end_plateau,
            slope_threshold=slope_threshold,
            min_length=min_length
        )

        # transition start
        rise_start_idx = get_rise_start_index(init_plateau)
        rise_t = x[rise_start_idx]

        # plateau start
        end_start_idx = end_plateau[0]
        end_t = x[end_start_idx]

        transition_t = end_t - rise_t
        plateau_value = np.mean(y_s[end_plateau[0]:end_plateau[1]])

        # midpoint
        midpoint_idx = math.floor((rise_start_idx + end_start_idx) / 2)
        midpoint_t = x[midpoint_idx]

        # overall slope
        dt = x[end_start_idx] - x[rise_start_idx]
        overall_slope = (y_s[end_start_idx] - y_s[rise_start_idx]) / dt if dt != 0 else np.nan

        # average local slope
        transition_x = x[rise_start_idx:end_start_idx]
        transition_y = y_s[rise_start_idx:end_start_idx]

        local_slopes = np.diff(transition_y) / np.diff(transition_x)
        avg_local_slope = np.mean(local_slopes)

    else:
        rise_start_idx = None
        midpoint_t = x[np.argmax(y_s)]
        plateau_value = None
        plateau_exists = 0
        overall_slope = None
        avg_local_slope = None
        transition_t = None


    auc = np.trapz(y_s, x)

    return {
        "Plateau Exists": plateau_exists,
        "Midpoint": midpoint_t,
        "Area Under Curve (AUC)": auc,
        "Average Local Slope": avg_local_slope,
        "Transition Overall Slope": overall_slope,
        "Transition Time": transition_t,
        "Plateau Value": plateau_value,
        "End Plateau": end_plateau,
        "Rise Start Idx": rise_start_idx
    }, (x, y, y_s)


def get_xy_dict(
        p_df: pd.DataFrame,
        bboxes: np.ndarray,
        mask_ID: int,
        mode: str,
        w_sg: int,
        poly: int,
        slope_threshold: float,
        min_length: int,
        min_value: float,
        search_start: int,
        T: float,
        p_thresh: float = 0.5,
        if_clustered: bool = False):
    """
    Classify droplets into X1/X0 and Y1/Y0 groups based on either:

    - plateau detection over time  (mode='plat')
    - final frame probability threshold (mode='p_thresh')

    Optionally, samples are further separated into clustered / isolated
    based on `is_clustered` at the reference frame `mask_ID`.
    """

    if mode not in ['plat', 'p_thresh']:
        raise KeyError("Mode must be either 'plat' or 'p_thresh'.")

    def get_stat(region, bbox):
        stats, (x, y, y_s) = get_curve_statistics(
            get_p(p_df, bbox, region),
            w_sg,
            poly,
            slope_threshold,
            min_length,
            min_value,
            search_start,
            T)

        return stats, x, y, y_s

    status_dict_inner = {}
    status_dict_outer = {}

    for bbox in bboxes:
        stats_inner, x_inner, y_inner, y_s_inner = get_stat('inner', bbox)
        stats_outer, x_outer, y_outer, y_s_outer = get_stat('outer', bbox)
        status_dict_inner[bbox] = (stats_inner, x_inner, y_inner, y_s_inner)
        status_dict_outer[bbox] = (stats_outer, x_outer, y_outer, y_s_outer)

    if if_clustered:
        df_mask = p_df.loc[p_df['frame_ID'] == mask_ID, ['bbox', 'is_clustered']]
        cluster_status = dict(zip(df_mask['bbox'], df_mask['is_clustered']))

        # default to False if key missing
        def is_clustered_fn(b):
            return bool(cluster_status.get(b, False))
    else:
        is_clustered_fn = lambda b: False  # dummy

    if mode == 'plat':
        X1 = [b for b in bboxes if status_dict_inner[b][0]['Plateau Exists']]
        Y1 = [b for b in bboxes if status_dict_outer[b][0]['Plateau Exists']]

    else:  # mode == 'p_thresh'

        final_frame = p_df['frame_ID'].max()
        df_final = p_df.loc[p_df['frame_ID'] == final_frame]

        X1 = df_final.loc[df_final['p_x'] > p_thresh, 'bbox'].tolist()
        Y1 = df_final.loc[df_final['p_y'] > p_thresh, 'bbox'].tolist()

    X0 = [b for b in bboxes if b not in X1]
    Y0 = [b for b in bboxes if b not in Y1]

    if not if_clustered:
        return {
            'X1': X1,
            'Y1': Y1,
            'X0': X0,
            'Y0': Y0
        }, (status_dict_inner, status_dict_outer)

    def split_clustered(bbox_list):
        clustered = [b for b in bbox_list if is_clustered_fn(b)]
        isolated = [b for b in bbox_list if not is_clustered_fn(b)]
        return clustered, isolated

    X1_clustered, X1_isolated = split_clustered(X1)
    Y1_clustered, Y1_isolated = split_clustered(Y1)

    clustered_all = [b for b in bboxes if is_clustered_fn(b)]
    isolated_all = [b for b in bboxes if not is_clustered_fn(b)]

    return {
        'X1 clustered': X1_clustered,
        'X1 isolated': X1_isolated,
        'Y1 clustered': Y1_clustered,
        'Y1 isolated': Y1_isolated,
        'X1': X1,
        'Y1': Y1,
        'X0': X0,
        'Y0': Y0,
        'clustered': clustered_all,
        'isolated': isolated_all,
    }, (status_dict_inner, status_dict_outer)


def one_zone_hard_soft(subset: pd.DataFrame,
                       frame_IDs) -> Tuple[List, List]:
    """
    Compute the time-resolved 'soft' and 'hard' corrosion fractions for a
    single region (inner X or outer Y).

    For droplets that ultimately belong to a given end-state set (X1 or Y1),
    this function evaluates at each frame t:

        soft(t) = mean( p(t) )
        hard(t) = n(p(t) > 0.5) / N

    where p(t) is either p_x or p_y depending on the subset.

    :param
    ----------
    subset : DataFrame
        Contains one row per droplet per frame, with:
            'frame_ID' : time index
            'p'        : probability for that region at that frame
    frame_IDs : iterable
        Ordered list of frame identifiers.

    :return
    -------
    soft : list of float
        Probability-weighted fraction vs time.
    hard : list of float
        Threshold-binary fraction vs time.
    """

    soft, hard = [], []

    for fid in frame_IDs:
        sf = subset[subset['frame_ID'] == fid]

        if len(sf) == 0:
            soft.append(np.nan)
            hard.append(np.nan)
            continue

        soft.append(sf['p'].mean())
        hard.append((sf['p'] > 0.5).mean())

    return soft, hard


def _safe_mean(series):
    """Return the mean of a pandas Series, or 0.0 if the Series is empty."""
    return series.mean() if len(series) > 0 else 0.0


def evolution_frac_per_frame(X1_bbox, Y1_bbox, X1Y1_bbox, subset):
    """
    Compute the time-dependent conditional co-occurrence fractions of inner
    (X) and outer (Y) region corrosion for a single frame, using both
    probability-weighted ("soft") and binary ("hard") definitions.

    This function implements Eq. (5) in the manuscript.

    Definitions
    -----------
    X1 :
        Set of droplets that exhibit end-state corrosion in the inner ROI.
    Y1 :
        Set of droplets that exhibit end-state corrosion in the outer ROI.
    X1Y1 :
        Intersection of X1 and Y1 — droplets that finally corrode in both ROIs.

    :param
    -----
    X1_bbox : list-like
        Bounding-box identifiers of droplets in X1.
    Y1_bbox : list-like
        Bounding-box identifiers of droplets in Y1.
    X1Y1_bbox : list-like
        Bounding-box identifiers of droplets in X1 ∩ Y1.
    subset : pd.DataFrame
        All droplet records corresponding to a *single frame*, containing:
            - 'bbox' : droplet identifier
            - 'p_x'  : corrosion probability in inner ROI at time t
            - 'p_y'  : corrosion probability in outer ROI at time t

    :return
    ------
    frac_x_soft : float
        Soft conditional co-occurrence fraction for X:
            frac_x_soft(t) = Kx * <p_xy>/<p_x>

    frac_y_soft : float
        Soft conditional co-occurrence fraction for Y:
            frac_y_soft(t) = Ky * <p_xy>/<p_y>

    frac_x_hard : float
        Hard conditional fraction based on threshold classification:
            frac_x_hard(t) = n_xy(t)/n_x(t)

    frac_y_hard : float
        Hard conditional fraction based on threshold classification:
            frac_y_hard(t) = n_xy(t)/n_y(t)

    Notes
    -----
    - The <·> averages are computed over droplets belonging to the FINAL
      STATE groups X1, Y1 and X1Y1, not frame-dependent groups.

    - Kx and Ky are *time-independent normalisation constants* obtained from
      end-state population ratios:

            Kx = N_XY / N_X
            Ky = N_XY / N_Y

      They arise from a linear-fit approximation linking the probability-
      weighted (soft) statistics to the empirically observed (hard)
      conditional fractions. This ensures soft and hard curves are on the
      same numerical scale.

    - Hard fractions use a binary decision rule p ≥ 0.5.

    - All divisions are protected against zero-division.

    """

    # Normalisation constants (linear-fit scaling)
    Kx = len(X1Y1_bbox) / len(X1_bbox) if len(X1_bbox) > 0 else 0.0
    Ky = len(X1Y1_bbox) / len(Y1_bbox) if len(Y1_bbox) > 0 else 0.0

    # Frame-restricted subsets
    sub_X1 = subset[subset['bbox'].isin(X1_bbox)]
    sub_Y1 = subset[subset['bbox'].isin(Y1_bbox)]
    sub_X1Y1 = subset[subset['bbox'].isin(X1Y1_bbox)]

    # Soft (probabilistic) statistics
    mean_pxy = _safe_mean(sub_X1Y1['p_x'] * sub_X1Y1['p_y'])
    mean_px  = _safe_mean(sub_X1['p_x'])
    mean_py  = _safe_mean(sub_Y1['p_y'])

    frac_x_soft = Kx * (mean_pxy / mean_px) if mean_px > 0 else 0.0
    frac_y_soft = Ky * (mean_pxy / mean_py) if mean_py > 0 else 0.0

    # Hard (threshold-based) statistics
    sub_xhard = subset[subset['p_x'] >= 0.5]
    sub_yhard = subset[subset['p_y'] >= 0.5]
    sub_xyhard = subset[(subset['p_x'] >= 0.5) & (subset['p_y'] >= 0.5)]

    frac_x_hard = len(sub_xyhard) / len(sub_xhard) if len(sub_xhard) > 0 else 0.0
    frac_y_hard = len(sub_xyhard) / len(sub_yhard) if len(sub_yhard) > 0 else 0.0

    return frac_x_soft, frac_y_soft, frac_x_hard, frac_y_hard


def get_x_y_binned(p_df, bin_edges, mask_ID, xy_dict, key: str):
    """
    Group droplets belonging to a chosen final–state set (X0/X1/Y0/Y1)
    into diameter bins, evaluated at the mask frame.

    :param
    ----------
    p_df:
        from Analysis.
    bin_edges : list of float
            Diameter bin boundaries (µm). If None, a single “full” bin is used.
    mask_ID: int
        ID used to generate the binary mask
    xy_dict : dict
        Mapping of state → bbox list, e.g. {'X1': [...], 'Y1': [...]}
    key : str
        One of {'X0','X1','Y0','Y1'} selecting the group.

    :return
    -------
    dia_dict : dict
        {bin → list of droplet diameters in that bin}
    bbox_dict : dict
        {bin → list of bbox IDs in that bin}
    count : dict
        {bin → number of droplets in that bin}
    """

    if key not in ['X0', 'X1', 'Y0', 'Y1']:
        raise KeyError("key must be one of {'X0','X1','Y0','Y1'}")

    df = p_df[p_df['frame_ID'] == mask_ID]
    df = df[df['bbox'].isin(xy_dict[key])]

    dia_dict, bbox_dict, count = {}, {}, {}

    # --- case 1: explicit bins ---
    if bin_edges:
        bin_edges = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-2)]
        for low, high in bin_edges:

            in_bin = df[
                (df['real_diameter'] > low) &
                (df['real_diameter'] <= high)
            ]

            dia_dict[(low, high)] = in_bin['real_diameter'].tolist()
            bbox_dict[(low, high)] = in_bin['bbox'].tolist()
            count[(low, high)] = len(in_bin)

    # --- case 2: single full group ---
    else:
        dia_dict['full'] = df['real_diameter'].tolist()
        bbox_dict['full'] = df['bbox'].tolist()
        count['full'] = len(df)

    return dia_dict, bbox_dict, count


def compute_cdf(data):
    """
    Return the empirical cumulative distribution function of data.
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

def per_bin_percentage(analyzed_dias, all_dias, bin_edges, ana_key):
    """
    Plot, per diameter bin, the proportion of droplets belonging to a
    given final-state group (X1 or Y1).

    :param
    ----------
    analyzed_dias : list
        Diameters of droplets belonging to X1 or Y1.
    all_dias : list
        Diameters of all droplets.
    bin_edges : list
        Diameter bin boundaries (µm).
    ana_key : {'X','Y'}
        Selects X1 or Y1.

    :return
    --------
    None.
    """

    frac_label = {'X': r'$X_{end}=1$ fraction', 'Y': r'$Y_{end}=1$ fraction'}
    bar_label = {'X': r'$X_{end}=1$', 'Y': r'$Y_{end}=1$'}

    all_hist, _ = np.histogram(all_dias, bins=bin_edges)
    ana_hist, _ = np.histogram(analyzed_dias, bins=bin_edges)

    frac = np.divide(
        ana_hist, all_hist,
        out=np.zeros_like(ana_hist, dtype=float),
        where=all_hist>0
    )

    bin_labels = [f"{bin_edges[i]}–{bin_edges[i+1]}"
                  for i in range(len(bin_edges)-1)]

    # Start plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar plot for B and A
    x = np.arange(len(bin_labels))  # position for each bin
    width = 0.4  # bar width

    ax1.bar(x, all_hist, width=width, label='All droplets', alpha=0.7)
    ax1.bar(x, ana_hist, width=width, label=bar_label[ana_key], alpha=0.7)

    ax1.set_xlabel(r'Droplet diameter ($\mu$m)', fontsize=16)
    ax1.set_ylabel('Count', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.tick_params(axis='both', labelsize=15)

    # Twin axis for fraction
    ax2 = ax1.twinx()
    line_fraction, = ax2.plot(x, frac, color='red', marker='o', label=frac_label[ana_key])
    ax2.set_ylabel('Fraction', fontsize=16)
    ax2.tick_params(axis='both', labelsize=15)

    lines_labels = [*ax1.get_legend_handles_labels()[0], line_fraction]
    labels = [*ax1.get_legend_handles_labels()[1], line_fraction.get_label()]
    ax1.legend(lines_labels, labels, fontsize=15)

def per_bin_percentage_both(analyzed_dias_x, analyzed_dias_y,
                            all_dias, bin_edges, mode='xy'):
    """
    Plot, per bin, the fraction of droplets belonging to
    X1 and Y1 (or corrosion types).

    mode = 'xy'   → X1 vs Y1
    mode = 'type' → Evans vs Only-outside
    """

    if mode == 'xy':
        frac_label = {'X': r'$X_{end}=1$ fraction',
                      'Y': r'$Y_{end}=1$ fraction'}
        bar_label = {'X': r'$X_{end}=1$',
                     'Y': r'$Y_{end}=1$'}
    elif mode == 'type':
        frac_label = {'X': 'Evans-like fraction',
                      'Y': 'Only-outside fraction'}
        bar_label = {'X': 'Evans-like',
                     'Y': 'Only outside'}
    else:
        raise KeyError

    all_hist, _ = np.histogram(all_dias, bins=bin_edges)
    x_hist, _ = np.histogram(analyzed_dias_x, bins=bin_edges)
    y_hist, _ = np.histogram(analyzed_dias_y, bins=bin_edges)

    frac_x = np.divide(x_hist, all_hist, out=np.zeros_like(x_hist,dtype=float), where=all_hist>0)
    frac_y = np.divide(y_hist, all_hist, out=np.zeros_like(y_hist,dtype=float), where=all_hist>0)

    bin_labels = [f"{bin_edges[i]}-{bin_edges[i + 1]}" for i in range(len(bin_edges) - 1)]
    x = np.arange(len(all_hist))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8,5))

    ax1.bar(x-width, x_hist, width=width, color='red', alpha=0.7, label=bar_label['X'])
    ax1.bar(x,       all_hist, width=width, color='orange', alpha=0.7, label='All')
    ax1.bar(x+width, y_hist, width=width, color='blue', alpha=0.7, label=bar_label['Y'])


    ax1.set_xlabel(r'Droplet diameter ($\mu$m)', fontsize=16)
    ax1.set_ylabel('Count', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.tick_params(axis='both', labelsize=15)

    ax2 = ax1.twinx()
    line_fraction_x, = ax2.plot(x, frac_x, color='red', marker='o', label=frac_label['X'], linewidth=4)
    line_fraction_y, = ax2.plot(x, frac_y, color='blue', marker='o', label=frac_label['Y'], linewidth=4)

    ax2.set_ylabel('Fraction', fontsize=16)
    ax2.tick_params(axis='both', labelsize=15)

    lines_labels = [*ax1.get_legend_handles_labels()[0], line_fraction_x, line_fraction_y]
    labels = [*ax1.get_legend_handles_labels()[1], line_fraction_x.get_label(), line_fraction_y.get_label()]
    ax1.legend(lines_labels, labels, fontsize=16, bbox_to_anchor=(0, 1.3), loc='upper left', ncol=2)


def collect_feature_and_curves_from_status(status_dict, bboxes, feature_name) -> Tuple[List, List, np.ndarray]:
    """
    Collect kinetic feature values and smoothed curves from stored status dicts.

    Parameters
    ----------
    status_dict: dict
        (status, x, y, y_s)
    bboxes : list
        Droplet identifiers.
    feature_name : str
        One of the feature names stored in stats dict.

    Returns
    -------
    feature_values : list of float
    smoothed_curves : list of np.ndarray
    time_axis : np.ndarray
        Same for all curves belonging to the region.
    """


    feature_values = []
    smoothed_curves = []
    time_axis = None

    for bbox in bboxes:
        stats, x, _, y_s = status_dict[bbox]

        value = stats.get(feature_name, None)
        if value is None:
            continue

        feature_values.append(value)
        smoothed_curves.append(y_s)

        if time_axis is None:
            time_axis = x

    return feature_values, smoothed_curves, time_axis

def mean_curve(curves):
    """
    Compute the mean curve across droplets (point-wise mean).
    """
    if len(curves) == 0:
        return None

    curves = np.vstack(curves)
    return curves.mean(axis=0)


class Analysis:
    def __init__(self,
                 mask_ID: int,
                 *,
                 w_sg: int = 23,
                 poly: int = 2,
                 slope_threshold: float = 0.005,
                 min_length: int = 5,
                 min_value: float = 0.5,
                 search_start: int = None,
                 T: float = 2.0):

        """
        Core analysis class.

        Analyse temporal curve of p to detect end-stage plateau behaviour
        and extract key kinetic indicators.

        The curve is first smoothed using a Savitzky–Golay filter. A late-stage plateau
        is detected as a contiguous region where the absolute slope remains below a
        threshold for at least `min_length` samples, and whose mean value exceeds
        `min_value`. The search for the end plateau begins at `search_start`
        (default = halfway point of the signal).

        If an end-stage plateau exists, the function also attempts to detect an
        early-stage (baseline) plateau that occurs before the rise and whose mean
        remains below the end-plateau level. If no such baseline plateau exists,
        the rise is assumed to begin near the start of the experiment
        (index = 2 by convention).

        Transition onset, plateau onset, midpoint timing, slopes and AUC are then
        computed from the smoothed curve.

        Plateaus are detected using the parameters below. These defaults are
        physically meaningful and consistent across the entire pipeline.

        :param
        ----------
        mask_ID: int
            frame_ID of the image to generate the binary mask.
        pos_ref_rgb: np.ndarray
            image for positional alignment.
        bri_ref_rgb: np.ndarray
            image for brightness alignment,
        w_sg : int, optional
            Window length for Savitzky–Golay smoothing (default = 23).
            Must be an odd integer ≥ polynomial order + 2.
        poly : int, optional
            Polynomial order for Savitzky–Golay smoothing (default = 2).
        slope_threshold : float, optional
            Maximum absolute gradient considered "flat" when detecting plateaus
            (default = 0.005).
        min_length : int, optional
            Minimum number of consecutive flat points required to define a plateau
            (default = 5).
        min_value : float, optional
            Minimum mean value of the late-stage plateau. This value is
            scientifically meaningful for your system and defaults to 0.5.
        search_start : int or None, optional
            Index from which to begin searching for an end-stage plateau.
            If None, defaults to half the curve length.
        T : float, optional
            Time resolution scaling factor. The time axis is defined as
            ``np.linspace(0, T * len(y), len(y))`` (default = 2.0 minutes per sample).

        Main public methods
        -------------------
        load_file(...)
            Load the pre-computed probability dataframe and compute geometry.
        compute_xy_dict(...)
            Perform plateau/threshold classification and compute per-droplet
            kinetic statistics for inner (X) and outer (Y) ROIs.

        Notes
        -----
        - All calculations are performed on the smoothed curve.
        - If no late-stage plateau is detected, transition-related quantities are None
          and the midpoint corresponds to the time of maximum signal.
        - If a baseline plateau cannot be detected, the rise onset is assumed to occur
          near the beginning (index = 2), consistent with the physical model.

        """
        self._p_df = None
        self._bbox = None
        self._xy_dict = None
        self._status_dict_inner = None
        self._status_dict_outer = None
        self.mask_ID = mask_ID

        # ---- Plateau detection configuration (GLOBAL for this object) ----
        self.w_sg = w_sg
        self.poly = poly
        self.slope_threshold = slope_threshold
        self.min_length = min_length
        self.min_value = min_value
        self.search_start = search_start
        self._T = T

    def load_file(self,
                  file_path: str = '',
                  um_per_pixel: float = (18.21 / 3.71) * (3.45 / 3),
                  thresh: float = None) -> None:
        """
        Load pre-computed df containing probability descriptors for all droplets in all frames.

        :param
        ----------
        file_path: str
            path of the pre-computed p_df file.
        um_per_pixel:
            the constant to convert unit pixel into um.
        thresh: float
            threshold of determining whether a droplet is considered as 'in a cluster' or 'isolated'.
            default = None
            if None is passed, thresh is automatically calculated from the median of each droplets' nearest neighbor distance.
            otherwise use the passed value.

        :return
        ----------
        None

        Effects
        ------------
        Updates the following attributes:

        self._p_df: pd.DataFrame

        self._bbox: np.ndarray

        """

        p_df = load_csv(file_path)
        p_df['real_diameter'] = 2 * np.sqrt(p_df['area'] / np.pi) * um_per_pixel  # um
        p_df['real_area'] = p_df['area'] * pow(um_per_pixel, 2)  # um^2
        p_df['diameter'] = 2 * np.sqrt(p_df['area'] / np.pi)  # pixel

        p_df = get_center(p_df)
        p_df = classify_droplet_spatial_clustering(self.mask_ID, p_df, thresh=thresh)

        self._p_df = p_df
        self._bbox = p_df['bbox'].unique()

    def compute_xy_dict(self,
                        mode: str,
                        p_thresh: float = 0.5,
                        if_clustered: bool = False
                        ) -> None:

        """
        Classify droplets into end-state inner-ROI (X1/X0) and outer-ROI (Y1/Y0)
        groups and compute per-droplet kinetic statistics.

        This method runs the full curve-analysis pipeline for all droplets in
        ``self.p_df`` using the plateau-detection parameters stored in the
        ``Analysis`` object. The results are stored internally as:

            self._xy_dict
            self._status_dict_inner
            self._status_dict_outer

        :param
        ----------
        mode : {'plat', 'p_thresh'}
            Selection rule for defining X1/Y1 membership:

            'plat' :
                A droplet is assigned to X1 (or Y1) if a late-stage plateau is
                detected in its inner-ROI (or outer-ROI) probability curve.

            'p_thresh' :
                A droplet is assigned to X1 (or Y1) if its final-frame probability
                exceeds ``p_thresh``.

        p_thresh : float, optional
            Probability threshold used only when ``mode='p_thresh'``.
            Default = 0.5.

        if_clustered : bool, optional
            If ``True``, droplets are further split into clustered / isolated
            sub-groups based on the ``is_clustered`` flag in ``p_df``.
            Additional dictionary keys are then created, e.g.:

                'X1 clustered', 'X1 isolated',
                'Y1 clustered', 'Y1 isolated'

        :return
        -------
        None.

        Notes
        -----
        - Plateau detection and kinetic feature extraction are performed by
          ``get_xy_dict`` / ``get_curve_statistics``.
        - Time axes are scaled using ``self._T``.
        - No plotting is performed here.

        Effects
        ------------
        Updates the following attributes:

        self._xy_dict : dict
            Mapping of droplet groups to bbox identifiers.

        self._status_dict_inner : dict
            Per-droplet kinetic statistics and smoothed curves for inner ROIs:
                {bbox : (stats_dict, x, y, y_s)}

        self._status_dict_outer : dict
            As above, for outer ROIs.

        """
        xy_dict, (status_dict_inner, status_dict_outer) = get_xy_dict(p_df=self._p_df,
                                                                      bboxes=self._bbox,
                                                                      mask_ID=self.mask_ID,
                                                                      mode=mode,
                                                                      w_sg=self.w_sg,
                                                                      poly=self.poly,
                                                                      slope_threshold=self.slope_threshold,
                                                                      min_length=self.min_length,
                                                                      min_value=self.min_value,
                                                                      search_start=self.search_start,
                                                                      T=self._T,
                                                                      p_thresh=p_thresh,
                                                                      if_clustered=if_clustered)

        self._xy_dict = xy_dict
        self._status_dict_inner = status_dict_inner
        self._status_dict_outer = status_dict_outer

    @property
    def p_df(self):
        if self._p_df is None:
            raise RuntimeError("p_df not updated yet. Call load_file() first.")
        return self._p_df

    @property
    def bbox(self):
        if self._bbox is None:
            raise RuntimeError("_bbox not updated yet. Call load_file() first.")
        return self._bbox

    @property
    def xy_dict(self):
        if self._bbox is None:
            raise RuntimeError("_p_df and _bbox not updated yet. Call load_file() first.")
        if self._xy_dict is None:
            raise RuntimeError("_xy_dict not updated yet. Call compute_xy_dict() first.")
        return self._xy_dict

    @property
    def status_dict_inner(self):
        if self._bbox is None:
            raise RuntimeError("_p_df and _bbox not updated yet. Call load_file() first.")
        if self._status_dict_inner is None:
            raise RuntimeError("_status_dict_inner not updated yet. Call compute_xy_dict() first.")
        return self._status_dict_inner

    @property
    def status_dict_outer(self):
        if self._bbox is None:
            raise RuntimeError("_p_df and _bbox not updated yet. Call load_file() first.")
        if self._status_dict_outer is None:
            raise RuntimeError("_status_dict_outer not updated yet. Call compute_xy_dict() first.")
        return self._status_dict_outer

    @property
    def T(self):
        return self._T


class Visualization:
    def __init__(self,
                 p_df: pd.DataFrame,
                 xy_dict: dict,
                 status_dict_inner: Dict,
                 status_dict_outer: Dict,
                 mask_ID: int,
                 pos_ref_rgb: np.ndarray,
                 bri_ref_rgb: np.ndarray,
                 T: float = 2.0
                 ):

        """
        Visualization toolbox for droplet corrosion probability analysis.

        This class operates on:
          - the processed dataframe (p_df)
          - droplet grouping dictionary (xy_dict)
          - inner/outer kinetic statistics (status_dict_inner/outer)

        No kinetic recomputation is performed here.

        :param
        ----------
        p_df: pd.DataFrame
            df obtained from Analysis.
        xy_dict: dict
            obtained from Analysis. Dictionary of droplet membership lists (bbox identifiers). Expected keys:
            -If `if_clustered is False`:
                'X1': bbox of droplets with end-state inner corrosion
                'Y1' : bbox of droplets with end-state outer corrosion
            -If `if_clustered is True`:
                'X1 clustered', 'X1 isolated'
                'Y1 clustered', 'Y1 isolated'
        status_dict_inner : dict of dict for inner ROIs, {bbox: (stats, x, y, y_s)}
            stats: Dict, containing:
                - 'Plateau Exists' : int (1 if a late-stage plateau is detected, else 0)
                - 'Midpoint' : float
                    Time corresponding to the midpoint between transition onset and
                    plateau onset (or time of maximum if no plateau exists).
                - 'Area Under Curve (AUC)' : float
                    Integral of the smoothed signal.
                - 'Average Local Slope' : float or None
                    Mean of local slopes within the transition region.
                - 'Transition Overall Slope' : float or None
                    Global slope between transition onset and plateau onset.
                - 'Transition Time' : float or None
                    Time difference between plateau onset and transition onset.
                - 'Plateau Value' : float or None
                    Mean value of the end-stage plateau. None if no plateau exists.
                - 'End Plateau' : tuple or None
                    (start_index, end_index) of the detected end plateau interval.
                    None if no plateau exists.
                - 'Rise Start Idx' : int or None
                    Index marking the end of the baseline plateau (or 2 if no baseline
                    plateau exists). None if no plateau is detected at all.
            x: ndarray
                Time axis corresponding to the signal.
            y: ndarray
                Original input signal
            y_s: ndarray
                Smoothed version of the input signal.
        status_dict_outer: dict of dict for outer ROIs,
            similar as status_dict_inner
        mask_ID: int
            frame_ID of the image to generate the binary mask.
        pos_ref_rgb: np.ndarray
            image for positional alignment.
        bri_ref_rgb: np.ndarray
            image for brightness alignment,
        T : float, optional
            Time resolution scaling factor. The time axis is defined as
            ``np.linspace(0, T * len(y), len(y))`` (default = 2.0 minutes per sample).

        Main public methods
        -------------------
        geo_stat_distribution(...)
            Fit and plot droplet size/aspect distributions.

        create_color_coding(...)
            Overlay inner/outer corrosion probability as a color heatmask
            on top of the droplet image.

        plot_curve_statistics(...)
            Plot temporal p-curves for a selected droplet with annotated
            transition / plateau behaviour.

        overall_x_y_feature(...)
            Plot histograms of kinetic features (midpoint, slope, etc.)
            for X1 and Y1 populations.

        x_y_hard_soft_label(...)
            Compare time-resolved soft vs. hard corrosion fractions and
            perform linear fits.

        overall_evolution_fraction(...)
            Plot time evolution of conditional co-corrosion fractions
            (Eq. 5 — soft and hard forms).

        x_y_final_hist(...)
            Plot final-state X0/X1/Y0/Y1 diameter distributions + CDF.

        evans_inter_final_hist(...)
            Plot population distributions for 'No corrosion' /
            'Evans-like' / 'Only outside' droplets.

        feature_hist_all_bin(...)
            For each diameter bin, plot kinetic feature distributions and
            mean probability-evolution curves.

        Notes
        -----
        This class assumes all preprocessing and classification are already
        performed by `Analysis`.

        """
        self._p_df = p_df
        self._xy_dict = xy_dict
        self._status_dict_inner = status_dict_inner
        self._status_dict_outer = status_dict_outer
        self.mask_ID = mask_ID
        self.pos_ref_rgb = pos_ref_rgb
        self.bri_ref_rgb = bri_ref_rgb
        self.if_clustered = True if 'clustered' in xy_dict.keys() else False
        self.T = T

    def geo_stat_distribution(self,
                              mode: str = 'size',
                              fitting_method: str = 'lognorm',
                              if_plot: bool = False) -> Dict[str, float]:
        """
        Compute population geometrical statistics.

        :param
        ----------
        mode: str
            Choose from 'size' and 'aspect'.
        fitting_method: str
            Choose from 'lognorm', 'norm' and 'gamma'.
        if_plot: bool
            whether to plot the distribution histogram and fitting or not.

        :return:
        fitting statitics
            - lognorm: mu and sigma
            - norm: mu and sigma
            - gamma: k and theta
        """


        if mode == 'size':
            dias = self._p_df[self._p_df['frame_ID'] == self.mask_ID]['real_diameter']
        elif mode == 'aspect':
            dias = self._p_df[self._p_df['frame_ID'] == self.mask_ID]['aspect_ratio']
        else:
            raise KeyError("Mode must be either 'size' or 'aspect'.")

        x = np.linspace(dias.min(), 800, 400) if mode == 'size' else np.linspace(dias.min(), dias.max(), 100)

        def norm_fitting(dias):
            mu, sigma = norm.fit(dias)
            p = norm.pdf(x, mu, sigma)

            if if_plot:
                plt.plot(x, p, 'r--', linewidth=2, label=fr"Normal fit ($\mu$={mu:.2f}, $\sigma$={sigma:.2f})")

            return {"mu": mu, "sigma": sigma}

        def lognorm_fitting(dias):
            s, loc, scale = lognorm.fit(dias, floc=0)
            mu = np.log(scale)
            sigma = s
            p = lognorm.pdf(x, s, loc=loc, scale=scale)

            if if_plot:
                plt.plot(x, p, 'g--', lw=2,
                         label=fr"Lognormal fit ($\mu$={mu:.2f}, $\sigma$={sigma:.2f})")

            return {"mu": mu, "sigma": sigma}

        def gamma_fitting(dias):
            k, loc, theta = gamma.fit(dias, floc=0)  # k=shape, theta=scale
            p = gamma.pdf(x, k, loc=loc, scale=theta)

            if if_plot:
                plt.plot(x, p, 'b:', lw=2,
                         label=fr"Gamma fit ($k$={k:.2f}, $\theta$={theta:.2f}; loc={loc:.2f})")

            return {"k": k, "theta": theta}

        if if_plot:
            sns.histplot(dias, stat="density", label="Droplet histogram", color="skyblue", kde=False)
            sns.kdeplot(dias, label="Data KDE", color="navy", lw=2)
            plt.legend()
            plt.xlabel("Droplet diameter (µm)") if mode == 'size' else plt.xlabel("Droplet aspect ratio")
            plt.ylabel("Density")

        if fitting_method == 'lognorm':
            return lognorm_fitting(dias)
        elif fitting_method == 'norm':
            return norm_fitting(dias)
        elif fitting_method == 'gamma':
            return gamma_fitting(dias)
        else:
            raise ValueError("Only 'lognorm' and 'norm' and 'gamma' are currently supported for fitting.")

    def create_color_coding(self,
                            source_ID: int,
                            source_imaeg_rgb: np.ndarray,
                            mask_coords_inner: Dict = None,
                            mask_coords_outer: Dict = None,
                            region: str = '',
                            mask_alpha: float = 0.5) -> None:

        """
        Overlay probability values onto a droplet image as a color-coded heatmask.

        This function visualizes either the inner-region probability (p_x) or
        outer-region probability (p_y) for all droplets in a given frame. For each
        droplet, the corresponding probability value is normalized to [0, 1] across
        that frame and mapped to a continuous colormap. The colored droplet regions
        are alpha-blended onto the brightness- and position-corrected source image.

        A second figure is produced showing only the probability heatmask.

        :param
        ----------
        source_ID : int
            Identifier of the frame to be visualized. This must match the
            'frame_ID' column in `self.p_df`.
        source_imaeg_rgb : np.ndarray
            RGB image array for the corresponding frame prior to normalization or
            alignment. Shape = (H, W, 3).
        mask_coords_inner : dict, optional
            Mapping from droplet bounding-box string to pixel coordinates belonging
            to the **inner** droplet region:
                { bbox_string : [[row, col], ...], ... }
            Must be provided when `region='inner'`.
        mask_coords_outer : dict, optional
            Mapping from droplet bounding-box string to pixel coordinates belonging
            to the **outer** region surrounding each droplet footprint.
            Must be provided when `region='outer'`.
        region : {'inner', 'outer'}
            Selects which probability field to visualize:
                'inner' → uses column 'p_x'
                'outer' → uses column 'p_y'
        mask_alpha: float
            blending ratio between base image and color coding.

        :return
        -------
        None.

        Notes
        -----
        - The source image is pre-processed for positional and brightness alignment
          using `image_preprocessing`.

        - Probabilities are **renormalized within the frame**:
              p_norm = (p - min) / (max - min)

        - The colormap used is `matplotlib.cm.coolwarm`.

        - Alpha blending is applied uniformly using:
              blended = base*(1−α) + mask*α
          with α = 0.5.

        - Mask pixels take on RGBA values while the base image remains RGB.

        - Two figures are produced:
             1) Base image + overlay mask
             2) Heatmask only
          both with a colorbar labelled "p Value (Corrosion intensity)".

        """

        if region == 'inner':
            region_key = 'p_x'
            if mask_coords_inner is None:
                raise ValueError("mask_coords_inner cannot be None if region == 'inner'")
            coords_passed = mask_coords_inner
        elif region == 'outer':
            region_key = 'p_y'
            if mask_coords_outer is None:
                raise ValueError("mask_coords_outer cannot be None if region == 'outer'")
            coords_passed = mask_coords_outer
        else:
            raise ValueError('region must be "inner" or "outer".')

        base_image = image_preprocessing(source_imaeg_rgb, self.pos_ref_rgb, self.bri_ref_rgb)
        image_height, image_width = base_image.shape[:-1]
        color_mask = np.zeros((image_height, image_width, 4), dtype=np.uint8)
        colormap = plt.get_cmap('coolwarm')
        norm = Normalize(vmin=0, vmax=1)
        frame_df = self._p_df[self._p_df['frame_ID'] == source_ID].copy()

        for index, row in frame_df.iterrows():
            bbox_id = row['bbox']
            px_value = row[f'{region_key}']

            # Get the pixel coordinates for this droplet
            if bbox_id in coords_passed:
                pixel_coords = np.array(coords_passed[bbox_id])
                rows, cols = pixel_coords[:, 0], pixel_coords[:, 1]

                # Convert the pX value to an RGBA color (values from 0-1)
                rgba_color = colormap(norm(px_value))

                # Convert color to 0-255 scale for the image array
                color_255 = tuple(int(c * 255) for c in rgba_color)

                # "Paint" the pixels on the mask with the corresponding color
                color_mask[rows, cols] = color_255

        base_image_float = base_image.astype(float) / 255.0

        # Extract the color (RGB) and transparency (Alpha) from the mask
        mask_rgb = color_mask[:, :, :3].astype(float) / 255.0

        blended_image_float = (base_image_float * (1 - mask_alpha)) + (mask_rgb * mask_alpha)

        blended_image = (blended_image_float * 255).astype(np.uint8)

        fig_1, ax_1 = plt.subplots(figsize=(8, 6))
        ax_1.imshow(blended_image)
        ax_1.axis('off')
        cbar_1 = fig_1.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=ax_1)
        cbar_1.set_label('p Value (Corrosion intensity)')

        fig_2, ax_2 = plt.subplots(figsize=(8, 6))
        ax_2.imshow(mask_rgb)
        ax_2.axis('off')

        cbar_2 = fig_2.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=ax_2)
        cbar_2.set_label('p Value (Corrosion intensity)')

    def plot_curve_statistics(self,
                              bbox: str,
                              region: str = ''
                              ) -> None:

        """
        Visualize temporal p curve statistics.

        :param
        ----------
        bbox : str
            1-D array of normalized signal values in the range [0, 1].
        region : str,
            choose from 'inner' and 'outer'
            Any other value raises an exception.

        :return
        -------
        None.

        """

        if region == 'inner':
            data_dict, x, y, y_s = self._status_dict_inner[bbox]
            axis_label = r'$p_X$'
        elif region == 'outer':
            data_dict, x, y, y_s = self._status_dict_outer[bbox]
            axis_label = r'$p_Y$'
        else:
            raise ValueError('region must be "inner" or "outer".')

        plateau_exists = data_dict["Plateau Exists"]
        midpoint_t = data_dict["Midpoint"]
        plateau_value = data_dict["Plateau Value"]
        end_plateau = data_dict["End Plateau"]
        rise_start_idx = data_dict["Rise Start Idx"]

        # Plotting the original and smoothed curves
        fig, ax1 = plt.subplots(figsize=(8, 6))

        plt.style.use("default")  # Ensure default style is used
        fig.patch.set_facecolor('white')  # White background for the figure
        ax1.set_facecolor('white')  # White background for the plot

        ax1.plot(x, y, label='Original', alpha=0.5, linewidth=4)
        ax1.plot(x, y_s, label='Smoothed', linewidth=4)

        if plateau_exists:
            ax1.axvline(x=midpoint_t, color='r', linestyle='--', label='Midpoint')
            ax1.hlines(plateau_value, xmin=x[end_plateau[0]], xmax=x[end_plateau[1] - 1],
                       colors='purple', linestyles='--', label='Plateau value')
        else:
            ax1.axvline(x=midpoint_t, color='g', linestyle='--', label='Time to Peak')

        # Highlight transition and plateau start points
        if rise_start_idx is not None and rise_start_idx < len(x):
            ax1.axvline(x=x[rise_start_idx], color='b', linestyle=':', label='Start of transition')
        if plateau_exists:
            ax1.axvline(x=x[end_plateau[0]], color='orange', linestyle=':', label='Start of plateau')

        ax1.tick_params(axis='both', direction='in', length=5, labelsize=16)
        plt.xlabel('Time (min)', fontsize=16)
        plt.ylabel(axis_label, fontsize=16)
        plt.legend(fontsize=15)
        plt.grid(False)
        plt.show()


    def overall_x_y_feature(self,
                            feature_name: str,
                            num_bins: int,
                            ) -> None:
        """
        Plot the distribution of a selected kinetic feature for droplets belonging
        to the X1 and Y1 transition groups, optionally split by clustered vs.
        isolated droplets.

        :param
        ----------
        feature_name : {'Average Local Slope',
                        'Transition Overall Slope',
                        'Midpoint',
                        'Transition Time'}
        num_bins : int
            Number of histogram bins.

        :return
        -------
        None.
        """

        if feature_name in ["Average Local Slope", "Transition Overall Slope"]:
            xaxis_label = "Slope"
        elif feature_name in ["Midpoint", "Transition Time"]:
            xaxis_label = "Time (min)"
        else:
            raise KeyError(f"Unsupported feature name: {feature_name}")

        if not self.if_clustered:

            X_vals = [v for bbox in self._xy_dict.get('X1', [])
                      if (v := self._status_dict_inner[bbox][0][feature_name]) is not None]

            Y_vals = [v for bbox in self._xy_dict.get('Y1', [])
                      if (v := self._status_dict_outer[bbox][0][feature_name]) is not None]

            sns.histplot(X_vals, bins=num_bins, kde=True, stat='density',
                         label='X', color='red')

            sns.histplot(Y_vals, bins=num_bins, kde=True, stat='density',
                         label='Y', color='blue')

        else:

            groups = {
                'X clustered': ('X1 clustered', 'X'),
                'X isolated': ('X1 isolated', 'X'),
                'Y clustered': ('Y1 clustered', 'Y'),
                'Y isolated': ('Y1 isolated', 'Y'),
            }

            for label, (dict_key, region) in groups.items():
                status_dict_ = self._status_dict_inner if region == 'X' else self._status_dict_outer
                vals = [v for bbox in self._xy_dict.get(dict_key, [])
                        if (v := status_dict_[bbox][0][feature_name]) is not None]

                sns.histplot(vals, bins=num_bins, kde=True, stat='density',
                             label=label)

        # ---------- Styling ----------
        plt.tick_params(axis='both', direction='in', length=5, labelsize=16)
        plt.xlabel(xaxis_label, fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=15)
        plt.grid(False)

    def x_y_hard_soft_label(self, region: str) -> None:
        """
        Plot the relationship between soft and hard corrosion fractions for
        inner-ROI (X), outer-ROI (Y), and co-occurring (XY) corrosion.

        For droplets belonging to the final-state sets X1 and Y1, this function
        computes, at each frame t:

            soft_X(t) = mean( p_X(t) )
            soft_Y(t) = mean( p_Y(t) )
            soft_XY(t) = mean( p_X(t) p_Y(t) )

            hard_X(t) = n( p_X(t) > 0.5 ) / n_X
            hard_Y(t) = n( p_Y(t) > 0.5 ) / n_Y
            hard_XY(t) = n( p_X>0.5 and p_Y>0.5 ) / n_XY

        These hard vs soft quantities are then compared through linear regression:

            hard ≈ m · soft + b

        which provides the proportionality factors k used in Eq. (5).

        :param
        ----------
        region: str
            choose from {'inner', 'outer', 'inner_outer'} for plotting

        :return
        ------
        None.

        Notes
        -----
        - The regression is empirical and performed on time-series samples.
        - Slopes m correspond to the scaling constants k in Eq. (5).
        """

        if region not in ['inner', 'outer', 'inner_outer']:
            raise ValueError("region must be one of 'inner', 'outer', 'inner_outer'.")

        df = self._p_df.copy()

        X1 = self._xy_dict['X1']
        Y1 = self._xy_dict['Y1']

        df['p_xy'] = df['p_x'] * df['p_y']

        subset_x = df[df['bbox'].isin(X1)].rename(columns={'p_x': 'p'})
        subset_y = df[df['bbox'].isin(Y1)].rename(columns={'p_y': 'p'})
        subset_xy = df[df['bbox'].isin(X1) & df['bbox'].isin(Y1)].rename(columns={'p_xy': 'p'})

        frame_IDs = self._p_df['frame_ID'].unique()

        soft_x, hard_x = one_zone_hard_soft(subset_x, frame_IDs)
        soft_y, hard_y = one_zone_hard_soft(subset_y, frame_IDs)
        soft_xy, hard_xy = one_zone_hard_soft(subset_xy, frame_IDs)

        x = np.linspace(1, self.T * len(frame_IDs), len(frame_IDs))

        # ---------- FIGURE 1: time-resolved ----------
        plt.figure()
        if region == 'inner':
            plt.plot(x, soft_x, color='red', label=r'$\overline{p_X}$')
            plt.plot(x, hard_x, color='red', linestyle='--', label=r'$Frac_X$')
        elif region == 'outer':
            plt.plot(x, soft_y, color='blue', label=r'$\overline{p_Y}$')
            plt.plot(x, hard_y, color='blue', linestyle='--', label=r'$Frac_Y$')
        else:
            plt.plot(x, soft_xy, color='green', label=r'$\overline{p_{XY}}$')
            plt.plot(x, hard_xy, color='green', linestyle='--', label=r'$Frac_{XY}$')

        plt.ylabel('p (or fraction)', fontsize=16)
        plt.xlabel('Time (min)', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.legend(fontsize=15)

        # ---------- FIGURE 2: regression ----------
        def safe_polyfit(xvals, yvals):
            xvals = np.array(xvals)
            yvals = np.array(yvals)
            mask = ~np.isnan(xvals) & ~np.isnan(yvals)
            return np.polyfit(xvals[mask], yvals[mask], 1)

        m_x, b_x = safe_polyfit(soft_x, hard_x)
        X_fit_x = np.linspace(min(soft_x), max(soft_x), 200)  # Smooth line
        X_fit_y = m_x * X_fit_x + b_x

        m_y, b_y = safe_polyfit(soft_y, hard_y)
        Y_fit_x = np.linspace(min(soft_y), max(soft_y), 200)  # Smooth line
        Y_fit_y = m_y * Y_fit_x + b_y

        m_xy, b_xy = safe_polyfit(soft_xy, hard_xy)
        XY_fit_x = np.linspace(min(soft_xy), max(soft_xy), 200)
        XY_fit_y = m_xy * XY_fit_x + b_xy

        plt.figure()
        if region == 'inner':
            plt.scatter(soft_x, hard_x, color='red', label='X')
            plt.plot(X_fit_x, X_fit_y, color='red', linestyle="--",
                     label=rf'$Frac^t_X={m_x:.2f}\overline{{p^t_X}}{b_x:.2f}$' if b_x < 0 else rf'$Frac_X={m_x:.2f}\overline{{p^t_X}}+{b_x:.2f}$')
        elif region == 'outer':
            plt.scatter(soft_y, hard_y, color='blue', label='Y')
            plt.plot(Y_fit_x, Y_fit_y, color='blue', linestyle="--",
                     label=rf'$Frac^t_Y={m_y:.2f}\overline{{p^t_Y}}{b_y:.2f}$' if b_y < 0 else rf'$Frac^t_Y={m_y:.2f}\overline{{p^t_Y}}+{b_y:.2f}$')
        else:
            plt.scatter(soft_xy, hard_xy, color='green', label='XY')
            plt.plot(XY_fit_x, XY_fit_y, color='green', linestyle="--",
                     label=rf'$Frac^t_{{XY}}={m_xy:.2f}\overline{{p^t_{{XY}}}}{b_xy:.2f}$' if b_xy < 0 else rf'$Frac_{{XY}}={m_xy:.2f}\overline{{p^t_{{XY}}}}+{b_xy:.2f}$')

        plt.xlabel(r'$\overline{p^t}$', fontsize=16)
        plt.ylabel('$Frac^t$', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.legend(fontsize=14, loc='upper left')

    def overall_evolution_fraction(self) -> None:
        """
        Plot the temporal evolution of the conditional co-occurrence fractions
        between inner-ROI (X) and outer-ROI (Y) corrosion, using both
        probability-weighted (“soft”) and binary (“hard”) definitions.

        This function is the time-resolved implementation associated with Eq. (5):
            soft fractions:
                f_X^soft(t) = K_X · <p_X(t)p_Y(t)> / <p_X(t)>
                f_Y^soft(t) = K_Y · <p_X(t)p_Y(t)> / <p_Y(t)>

            hard fractions:
                f_X^hard(t) = n_XY(t) / n_X(t)
                f_Y^hard(t) = n_XY(t) / n_Y(t)

        where the angular brackets denote averages taken over droplets belonging
        to the *final-state* sets X1, Y1 and X1Y1.

        :param
        ----------
        None.

        :return
        ------
        None

        Behaviour
        ---------
        - For each frame, the function calls `evolution_frac_per_frame`, which
          implements Eq. (5) to obtain:

              frac_x_soft(t), frac_y_soft(t), frac_x_hard(t), frac_y_hard(t)

        - If droplets are further separated by spatial clustering state, the soft
          fractions are computed and plotted separately for clustered and isolated
          subsets.

        - Soft curves are plotted as solid lines (probability-weighted).
          Hard curves are plotted as dashed lines (thresholded binary).

        Notes
        -----
        - The normalisation constants K_X and K_Y (used inside
          `evolution_frac_per_frame`) are time-independent and depend only on
          end-state population ratios. They arise from a linear-fit approximation
          linking soft and hard statistics.

        - The time coordinate is converted externally to minutes; here the index
          simply follows the order of `frame_IDs`.

        - All divisions are protected against zero-division inside
          `evolution_frac_per_frame`.

        """
        frame_IDs = self._p_df['frame_ID'].unique()

        def get_fraction_series(X1, Y1):
            """Compute soft and hard fraction time-series for a given X1/Y1 set."""
            X1Y1 = [b for b in X1 if b in Y1]

            x_soft, x_hard = [], []
            y_soft, y_hard = [], []

            for fid in frame_IDs:
                subset = self._p_df[self._p_df['frame_ID'] == fid]
                fx_s, fy_s, fx_h, fy_h = evolution_frac_per_frame(
                    X1, Y1, X1Y1, subset
                )
                x_soft.append(fx_s)
                y_soft.append(fy_s)
                x_hard.append(fx_h)
                y_hard.append(fy_h)

            return x_soft, x_hard, y_soft, y_hard

        # Time axis
        x = np.linspace(1, self.T * len(frame_IDs), len(frame_IDs))

        if not self.if_clustered:

            X1 = self._xy_dict['X1']
            Y1 = self._xy_dict['Y1']

            X_soft, X_hard, Y_soft, Y_hard = get_fraction_series(X1, Y1)

            plt.plot(x, X_soft, color='red', label=r'$k_X\cdot\frac{\overline{p^t_{XY}}}{\overline{p_X^t}}$')
            plt.plot(x, X_hard, color='red', linestyle='--', label=r'$\frac{n^t_{XY}}{n^t_X}$')

            plt.plot(x, Y_soft, color='blue', label=r'$k_Y\cdot\frac{\overline{p^t_{XY}}}{\overline{p_Y^t}}$')
            plt.plot(x, Y_hard, color='blue', linestyle='--', label=r'$\frac{n^t_{XY}}{n^t_Y}$')

        else:
            X1_c = self._xy_dict['X1 clustered']
            X1_i = self._xy_dict['X1 isolated']
            Y1_c = self._xy_dict['Y1 clustered']
            Y1_i = self._xy_dict['Y1 isolated']

            Xc_soft, Xc_hard, Yc_soft, Yc_hard = get_fraction_series(X1_c, Y1_c)
            Xi_soft, Xi_hard, Yi_soft, Yi_hard = get_fraction_series(X1_i, Y1_i)

            # Plot soft fractions only
            plt.plot(x, Xc_soft, color='red',
                     label='Clustered X')  # label=r'Clustered: $k_X\cdot\frac{\overline{p^t_{XY}}}{\overline{p_X^t}}$')
            #plt.plot(x, Xc_hard, color='red',linestyle='--', label=r'Clustered: $\frac{n^t_{XY}}{n^t_X}$')

            plt.plot(x, Xi_soft, color='orange',
                     label='Isolated X')  # label=r'Isolated: $k_X\cdot\frac{\overline{p^t_{XY}}}{\overline{p_X^t}}$')
            #plt.plot(x, Xi_hard, color='orange',linestyle='--', label=r'Isolated: $\frac{n^t_{XY}}{n^t_X}$')

            plt.plot(x, Yc_soft, color='blue',
                     label='Clustered Y')  # label=r'Clustered: $k_Y\cdot\frac{\overline{p^t_{XY}}}{\overline{p_Y^t}}$')
            #plt.plot(x, Yc_hard, color='blue',linestyle='--', label=r'Clustered: $\frac{n^t_{XY}}{n^t_Y}$')

            plt.plot(x, Yi_soft, color='green',
                     label='Isolated Y')  # label=r'Isolated: $k_Y\cdot\frac{\overline{p^t_{XY}}}{\overline{p_Y^t}}$')
            #plt.plot(x, Yi_hard, color='green',linestyle='--', label=r'Isolated: $\frac{n^t_{XY}}{n^t_Y}$')

        plt.tick_params(labelsize=16)
        plt.legend(fontsize=15, ncol=2)
        plt.ylabel('Fraction', fontsize=16)
        plt.xlabel('Time (min)', fontsize=16)

    def x_y_final_hist(self, num_bins, bin_edges) -> None:
        """
        Plot droplet diameter statistics for the four end-state classes:

            • X0 : no inner-ROI corrosion at end state
            • X1 : inner-ROI corrosion at end state
            • Y0 : no outer-ROI corrosion at end state
            • Y1 : outer-ROI corrosion at end state

        Two subplots are generated:

            (1) Probability-density histograms (with KDE overlay)
            (2) Empirical cumulative distribution functions (CDFs)

        In addition, a per-diameter-bin plot is produced showing the fraction
        of droplets belonging to X1 and Y1 relative to the full population.

        :param
        ----------
        num_bins : int
            Number of histogram bins used for the diameter distributions.

        bin_edges : list of float
            Diameter bin boundaries (µm) supplied to
            ``per_bin_percentage_both`` for computing the per-bin X1 and Y1
            fractions.

        :return
        -------
        None
            Figures are produced but no data are returned.

        Behaviour
        ---------
        - X and Y denote inner- and outer-ROI corrosion, respectively.
        - The droplet class membership (X0/X1/Y0/Y1) is taken from
          ``self._xy_dict``.
        - Droplet diameters are taken from ``'real_diameter'`` at the mask
          frame ``self.mask_ID``.
        - KDE smoothing is applied purely for visualisation.

        Notes
        -----
        - Assumes ``Analysis.compute_xy_dict`` has already been called.
        - The supplementary per-bin fraction plot is generated internally via
          ``per_bin_percentage_both(..., mode='xy')``.
        """
        key_name = {'X0': r'$X_{end}=0$', 'X1': r'$X_{end}=1$',
                    'Y0': r'$Y_{end}=0$', 'Y1': r'$Y_{end}=1$'}
        colors = {'X0': 'pink', 'X1': 'red', 'Y0': 'skyblue', 'Y1': 'blue'}

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        for key, bboxes in self._xy_dict.items():
            vals = self._p_df[
                (self._p_df['bbox'].isin(bboxes)) &
                (self._p_df['frame_ID'] == self.mask_ID)
                ]['real_diameter'].tolist()

            sns.histplot(vals, bins=num_bins, kde=True, stat='density',
                         label=key_name[key], color=colors[key], ax=axs[0])

            sorted_data, cdf = compute_cdf(vals)
            axs[1].plot(sorted_data, cdf, label=key_name[key], color=colors[key])

        axs[0].legend(fontsize=16)
        axs[0].tick_params(axis='both', labelsize=15)
        axs[0].set_xlabel(r'Droplet diameter ($\mu$m)$', fontsize=16)
        axs[0].set_ylabel('Density', fontsize=16)

        axs[1].legend(fontsize=16)
        axs[1].tick_params(axis='both', labelsize=15)
        axs[1].set_xlabel(r'Droplet diameter ($\mu$m)$', fontsize=16)
        axs[1].set_ylabel('Cumulative density', fontsize=16)

        # Overlay X1/Y1 per-bin fractions
        all_dia = self._p_df[self._p_df['frame_ID'] == self.mask_ID]['real_diameter'].tolist()
        X1_dia = self._p_df[self._p_df['bbox'].isin(self._xy_dict['X1']) &
                           (self._p_df['frame_ID'] == self.mask_ID)]['real_diameter'].tolist()
        Y1_dia = self._p_df[self._p_df['bbox'].isin(self._xy_dict['Y1']) &
                           (self._p_df['frame_ID'] == self.mask_ID)]['real_diameter'].tolist()

        per_bin_percentage_both(X1_dia, Y1_dia, all_dia,
                                     bin_edges=bin_edges, mode='xy')

    def evans_inter_final_hist(self, num_bins, bin_edges) -> None:
        """
        Plot droplet diameter distributions and CDFs for the three corrosion
        outcome classes:

            • No corrosion      :  X0 ∩ Y0
            • Evans-like        :  X1
            • Only-outside      :  Y1 ∩ X0

        where X and Y denote inner-ROI and outer-ROI corrosion states,
        respectively.

        Two panels are generated:

            (1) Probability-density histograms (with KDE overlay)
            (2) Empirical cumulative distribution functions (CDFs)

        A third overlay plot is produced showing, per-diameter bin, the fraction
        of droplets belonging to the Evans-like and Only-outside classes.

        :param
        ----------
        num_bins : int
            Number of histogram bins for the diameter distributions.

        bin_edges : list of float
            Diameter bin boundaries (µm) used for computing the per-bin
            corrosion-type fractions.

        :return
        -------
        None

        Behaviour
        ---------
        - 'No corrosion' droplets are those that never reach end-state in either ROI.
        - 'Evans-like' droplets reach end-state in the inner ROI (X1).
        - 'Only-outside' droplets reach end-state in the outer ROI but not the inner
          ROI (Y1 ∩ X0).

        - All statistics are evaluated at the mask frame ``self.mask_ID``.
        - The helper function ``per_bin_percentage_both`` is called internally to
          compute and plot the per-bin fractions.


        Notes
        -----
        - This function assumes ``self._xy_dict`` has already been computed using
          ``Analysis.compute_xy_dict``.
        - Droplet diameters are taken from ``'real_diameter'`` in ``self._p_df``.
        """
        bbox_no_corr = [i for i in self._xy_dict['X0'] if i in self._xy_dict['Y0']]
        bbox_evans = self._xy_dict['X1']
        bbox_inter = [i for i in self._xy_dict['Y1'] if i in self._xy_dict['X0']]

        type_dict = {'No': bbox_no_corr, 'Evans': bbox_evans, 'Inter': bbox_inter}
        color_dict = {'No': 'green', 'Evans': 'red', 'Inter': 'blue'}
        key_name_dict = {'No': r'No corrosion', 'Evans': r'Evans-like', 'Inter': r'Only outside'}

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        subset_evans_dia = \
        self._p_df[(self._p_df['bbox'].isin(bbox_evans)) & (self._p_df['frame_ID'] == self.mask_ID)][
            'real_diameter'].tolist()
        subset_outward_dia = \
        self._p_df[(self._p_df['bbox'].isin(bbox_inter)) & (self._p_df['frame_ID'] == self.mask_ID)][
            'real_diameter'].tolist()
        subset_all = self._p_df[(self._p_df['frame_ID'] == self.mask_ID)]['real_diameter'].tolist()

        for key, valid_bboxes in type_dict.items():
            subset_type_dia = self._p_df[
                (self._p_df['bbox'].isin(valid_bboxes)) & (self._p_df['frame_ID'] == self.mask_ID)][
                'real_diameter'].tolist()
            sns.histplot(subset_type_dia, bins=num_bins, kde=True, stat='density', label=key_name_dict[key],
                         color=color_dict[key], ax=axs[0])
            # cdf
            sorted_data, cdf = compute_cdf(subset_type_dia)
            axs[1].plot(sorted_data, cdf, label=key_name_dict[key], color=color_dict[key])

        axs[0].legend(fontsize=16)
        axs[0].tick_params(axis='both', labelsize=15)
        axs[0].set_xlabel(r'Droplet diameter ($\mu$m)', fontsize=16)
        axs[0].set_ylabel('Density', fontsize=16)

        axs[1].legend(fontsize=16)
        axs[1].tick_params(axis='both', labelsize=15)
        axs[1].set_xlabel(r'Droplet diameter ($\mu$m)', fontsize=16)
        axs[1].set_ylabel('Cumulative density', fontsize=16)

        per_bin_percentage_both(subset_evans_dia, subset_outward_dia, subset_all, bin_edges=bin_edges, mode='type')

    def feature_hist_all_bin(
            self,
            bin_edges,
            feature_name,
            key,
            num_bins=10
    ) -> Tuple[Dict, Dict]:
        """
        For each diameter bin, plot:

            (1) histogram of a selected kinetic feature
            (2) average temporal evolution curve

        using pre-computed stats from status_dict_inner / status_dict_outer.

        :param
        ----------
        bin_edges : list of (low, high) tuples or None
        feature_name : str
            Feature key stored in stats dict.
        key : str
            One of {'X0','X1','Y0','Y1'} selecting droplet group.
        num_bins : int
            Histogram bins.

        """

        if key not in ['X0', 'X1', 'Y0', 'Y1']:
            raise ValueError("Key must in 'X0', 'X1', 'Y0', 'Y1'.")

        region = 'X' if key.startswith('X') else 'Y'
        status_dict = self._status_dict_inner if region == 'X' else self._status_dict_outer
        axis_label = r'$p_X$' if region == 'X' else r'$p_Y$'
        if feature_name == "Average Local Slope":
            xaxis_label = 'Slope'
        elif feature_name == "Transition Overall Slope":
            xaxis_label = 'Slope'
        elif feature_name == "Midpoint":
            xaxis_label = 'Time (min)'
        elif feature_name == "Transition Time":
            xaxis_label = 'Time (min)'
        else:
            raise ValueError("Feature name must be 'Average Local Slope', 'Transition Overall Slope', 'Midpoint', 'Transition Time'.")


        # ---- group droplets into bins ----
        _, bbox_dict, _ = get_x_y_binned(self._p_df, bin_edges, self.mask_ID, self._xy_dict, key)

        all_bin_feature = {}
        all_bin_avg_curve = {}

        colors = (
            [plt.cm.coolwarm(i / (len(bbox_dict) - 1)) for i in range(len(bbox_dict))]
            if bin_edges else ['red']
        )


        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        for i, (bound, bboxes) in enumerate(bbox_dict.items()):

            feature_vals, curves, time_x = collect_feature_and_curves_from_status(status_dict, bboxes, feature_name)

            mean_curve_ = mean_curve(curves)

            all_bin_feature[bound] = feature_vals
            all_bin_avg_curve[bound] = mean_curve_


            label = (
                f"{bound[0]}–{bound[1]} µm"
                if isinstance(bound, tuple)
                else "Full range"
            )

            sns.histplot(feature_vals,
                         bins=num_bins,
                         stat='density',
                         kde=True,
                         ax=axs[0],
                         label=label,
                         color=colors[i])

            axs[1].plot(time_x, mean_curve_,
                        label=label,
                        color=colors[i],
                        linewidth=3)

            axs[0].legend(fontsize=15)
            axs[0].set_xlabel(xaxis_label, fontsize=16)
            axs[0].set_ylabel("Density", fontsize=16)
            axs[0].tick_params(labelsize=15)

            axs[1].legend(fontsize=15)
            axs[1].set_xlabel("Time (min)", fontsize=16)
            axs[1].set_ylabel(axis_label, fontsize=16)
            axs[1].tick_params(labelsize=15)

        return all_bin_feature, all_bin_avg_curve

