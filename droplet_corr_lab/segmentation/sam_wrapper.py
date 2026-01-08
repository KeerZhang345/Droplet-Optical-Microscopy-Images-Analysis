import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import segment_anything.automatic_mask_generator
from torchvision.ops import boxes as box_ops

# Save the original batched_nms function
_orig_batched_nms = box_ops.batched_nms

def batched_nms_patched(boxes, scores, idxs, iou_threshold):
    # Ensure the category indices (idxs) match the device of the boxes
    if boxes.device != idxs.device:
        idxs = idxs.to(boxes.device)
    return _orig_batched_nms(boxes, scores, idxs, iou_threshold)

# Apply the patch to the specific module SAM uses
segment_anything.automatic_mask_generator.batched_nms = batched_nms_patched

from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from multiprocessing import get_context


def draw_rectangle(ax, anns):
    """
    Draw bounding-box rectangles onto a transparent RGBA overlay
    and render it on the provided Matplotlib axis.

    :param
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw.

    anns : list of dict
        SAM annotation list. Each dict must contain:
            'bbox' : [x, y, w, h]
            'segmentation' : 2-D boolean mask

    :return
    ----------
    None.
    """
    if anns:
        # Sort largest→smallest
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        H, W = anns[0]['segmentation'].shape
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        for ann in anns:
            x, y, w, h = ann['bbox']
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))

            cv2.rectangle(
                overlay,
                pt1,
                pt2,
                color=(0, 0, 255, 255),  # RGBA blue
                thickness=2
            )

        ax.imshow(overlay.astype(np.float32) / 255.0)


def show_anns(ax, anns, color=True):
    """
    Visualize SAM segmentation masks on a Matplotlib axis.

    :param
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw.

    anns : list of dict
        SAM annotations, each containing key 'segmentation'.

    color : bool
        If True  → overlay random RGBA colors
        If False → return a binary mask image

    :return
    -------
    np.ndarray
        RGBA overlay (color=True)
        or
        2-D binary mask (color=False)
    """

    if not anns:
        return

    anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    H, W = anns[0]['segmentation'].shape

    if color:
        img = np.zeros((H, W, 4), dtype=np.float32)
    else:
        img = np.zeros((H, W), dtype=np.uint8)

    for ann in anns:
        mask = ann['segmentation']

        if color:
            rgba = np.concatenate([np.random.rand(3), [0.35]])
            img[mask] = rgba
        else:
            img[mask] = 1

    ax.imshow(img)
    return img


def filter_masks_heuristic(
        masks: List[dict],
        max_area_ratio: float = 0.2,
        aspect_threshold: float = 10.0,
        min_solidity: float = 1.0,
        overlap_threshold: float = 0.5,
        neighbor_size_range: Tuple[float, float] = (0.2, 0.8),
        safe_size_threshold: float = 0.1,
        max_overlap_count: int = 2
) -> List[dict]:
    """
    Refines SAM segmentation masks using geometric heuristics and spatial constraints.

    This filter operates in two phases to remove artifacts while preserving "troublesome"
    larger areas if they are statistically significant.

    Phase 1: Geometric Pre-filtering (Shape Analysis)
    -------------------------------------------------
    Removes unstable shapes likely to be noise, cracks, or shadows.
    * **Contour Analysis:** Calculates the contour and convex hull of every mask.
    * **Solidity Check:** If a shape is highly elongated (high aspect ratio) AND
        non-convex (low solidity, i.e., much smaller than its hull), it is discarded.
        This effectively targets thin, squiggly artifacts.

    Phase 2: Vectorized Spatial Conflict (NMS-like Logic)
    -----------------------------------------------------
    Removes redundant masks based on overlap, specifically targeting fragmentation
    where a large object is split into many smaller sub-masks.

    * **Safety Guard:** Masks larger than `safe_size_threshold` are "anchors" and
        are NEVER removed, regardless of overlaps.
    * **Neighbor Selection:** For a mask `i`, we only look for conflict with neighbors `j`
        that satisfy:
        1. `Area(i) > Area(j)` (We only penalize the larger mask for having smaller debris inside?)
           *Note based on original logic: This implies we want to remove `i` if it covers many `j`s.*
        2. `Area(j)` is within `neighbor_size_range` relative to `Area(i)`.
    * **Overlap Check:** If mask `i` overlaps with more than `max_overlap_count`
        neighbors (based on `overlap_threshold`), mask `i` is discarded.

    :param
    ----------
    masks : List[dict]
        SAM output list. Each dict must have 'segmentation' and 'bbox'.

    max_area_ratio: float
        Area above this threshold will be removed (possible background region)

    aspect_threshold : float
        Max aspect ratio (long/short) allowed before the solidity check is triggered.
        (e.g., 3.0 means masks 3x longer than wide get checked).

    min_solidity : float
        Minimum solidity (contour_area / hull_area) required
        for elongated masks. Low solidity implies a "squiggly" or empty shape.

    overlap_threshold : float
        Intersection-Over-Area ratio required to count a
        neighbor as a "conflict".

    neighbor_size_range : Tuple[float, float]
        The size window for valid neighbors.
        A neighbor `j` is only considered if `low <= Area(j)/Area(i) <= high`.

    safe_size_threshold : float
        Masks larger than this fraction of the image
        (0.0-1.0) are exempt from removal logic.

    max_overlap_count : int
        Max number of valid overlapping neighbors allowed before the mask is removed.
        (Default 2).

    :return
    -------
    List[dict]
        The filtered list of mask dictionaries.
    """

    # --- PHASE 1: Geometric Pre-filtering ---
    candidates = []

    for mask in masks:
        m = mask['segmentation']
        mask_area = np.sum(m)
        canvas_area = m.shape[0] * m.shape[1]

        # 1. Discard huge artifacts
        if mask_area > max_area_ratio * canvas_area:
            continue

        # 2. Shape Analysis (Contours)
        formatted_mask = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(formatted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            continue

        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Check Aspect Ratio & Solidity
        if len(approx) > 0:
            rect = cv2.minAreaRect(approx)
            w_rect, h_rect = rect[1]

            if min(w_rect, h_rect) > 0:
                ar = max(w_rect, h_rect) / min(w_rect, h_rect)

                # If shape is elongated, it must be "solid" (convex)
                if ar > aspect_threshold:
                    c_area = cv2.contourArea(contours[0])
                    hull_area = cv2.contourArea(cv2.convexHull(contours[0]))

                    if hull_area == 0 or (c_area / hull_area) < min_solidity:
                        continue
            else:
                continue

        # Cache values for Phase 2
        mask['area_val'] = mask_area
        mask['canvas_val'] = canvas_area
        candidates.append(mask)

    if not candidates:
        return []

    # --- PHASE 2: Spatial Conflict (Vectorized) ---

    boxes = np.array([c['bbox'] for c in candidates], dtype=np.float32)
    areas = np.array([c['area_val'] for c in candidates], dtype=np.float32)

    # Pre-calculate Box coords (x1, y1, x2, y2)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]

    final_masks = []
    low_ratio, high_ratio = neighbor_size_range

    for i, mask_i in enumerate(candidates):
        area_i = areas[i]

        # Rule: If mask is "Safe" (large relative to canvas), skip penalty logic
        if area_i >= safe_size_threshold * mask_i['canvas_val']:
            final_masks.append(mask_i)
            continue

        # 1. Find neighbors 'j' where:
        #    - area_i > area_j (Only check smaller neighbors)
        #    - area_j is within size range of area_i
        valid_targets = (areas < area_i) & \
                        (areas >= area_i * low_ratio) & \
                        (areas <= area_i * high_ratio)

        if not np.any(valid_targets):
            final_masks.append(mask_i)
            continue

        target_indices = np.where(valid_targets)[0]

        # 2. Vectorized BBox Overlap
        xx1 = np.maximum(x1[i], x1[target_indices])
        yy1 = np.maximum(y1[i], y1[target_indices])
        xx2 = np.minimum(x2[i], x2[target_indices])
        yy2 = np.minimum(y2[i], y2[target_indices])

        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        inter_area = w_inter * h_inter

        # Threshold: intersection > overlap_threshold * neighbor_area
        bbox_thresh = overlap_threshold * areas[target_indices]

        bbox_pass_mask = inter_area > bbox_thresh
        target_indices = target_indices[bbox_pass_mask]

        if len(target_indices) == 0:
            final_masks.append(mask_i)
            continue

        # 3. Precise Segmentation Overlap (Lazy Evaluation)
        overlap_count = 0
        seg_i = mask_i['segmentation']

        for j in target_indices:
            seg_j = candidates[j]['segmentation']

            # Fast boolean AND
            pixel_overlap = np.sum(seg_i & seg_j)

            if pixel_overlap > overlap_threshold * areas[j]:
                overlap_count += 1
                if overlap_count > max_overlap_count:
                    break

        if overlap_count <= max_overlap_count:
            final_masks.append(mask_i)

    return final_masks


def refine_one_patch(args):
    (r, c), patch_rgb, anns_raw, filter_kwargs = args

    anns_filtered = filter_masks_heuristic(
        masks=anns_raw,
        **filter_kwargs
    )

    return ((r, c), patch_rgb, anns_filtered)


def visualize_mask_grid_place_aware(
    patches,
    num_rows,
    num_cols,
):
    """
    patches: list of ((row, col), image_rgb, anns)
    anns = list of SAM dict outputs
    """

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4 * num_cols, 4 * num_rows)
    )

    axs = np.array(axs)

    # Initialize grid with black placeholders
    for r in range(num_rows):
        for c in range(num_cols):
            axs[r, c].imshow(np.zeros((10, 10, 3)))
            axs[r, c].axis("off")

    # Fill the true patch locations
    for (r, c), image_rgb, anns in patches:

        ax = axs[r, c]
        ax.imshow(image_rgb)
        draw_rectangle(ax=ax, anns=anns)
        show_anns(ax=ax, anns=anns, color=True)
        ax.axis('off')
        """for ann in anns:
            mask = ann['segmentation']
            rgba = np.concatenate([np.random.rand(3), [0.35]])

            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = rgba
            ax.imshow(overlay)
            draw_rectangle(ax=ax, anns=mask)"""

    plt.tight_layout()
    plt.show()


class SAMMaskGenerator:
    """
    A wrapper around the Segment Anything Model (SAM) for automated mask generation
    on image patches. Handles model initialization, primary segmentation, and
    heuristic refinement of masks.
    """
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        points_per_side: int = 70,
        pred_iou_thresh: float = 0.75,
        stability_score_thresh: float = 0.75,
        min_mask_region_area: int = 50,
    ):
        """
        Initialize the SAM model and the automatic mask generator.

        :param
        ----------
        checkpoint_path : str
            Path to the model checkpoint file (.pth).

        model_type : str, optional
            The type of model to load (e.g., 'vit_h', 'vit_l', 'vit_b').
            Default is "vit_h".

        points_per_side : int, optional
            The number of points to sample along one side of the image.
            The total number of points is points_per_side**2.
            Higher numbers detect smaller objects but are slower. Default is 70.

        pred_iou_thresh : float, optional
            A filtering threshold in [0,1]. The model's own prediction of mask quality.
            Masks with predicted IoU < thresh are discarded. Default is 0.75.

        stability_score_thresh : float, optional
            A filtering threshold in [0,1]. Measures mask stability by checking
            IoU between the mask binarized at different thresholds. Default is 0.75.

        min_mask_region_area : int, optional
            If > 0, post-processing will remove disconnected mask regions or holes
            smaller than this area (in pixels). Default is 50.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("PyTorch version:", torch.__version__)
        print("CUDA is available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())
        print(torch.cuda.get_device_name()) if torch.cuda.is_available() else print("No cuda available.")

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
        )

        print(f"SAM model loaded on: {self.device}")

    def segment_primary(
        self,
        patches_rgb: List[Tuple[Tuple[int, int], np.ndarray]],
        num_rows: int,
        num_cols: int,
        save_dir: str,
        mask_ID: int,
        ) -> List[Tuple[Tuple[int, int], np.ndarray, List[dict]]]:

        """
        Runs the raw SAM inference on a list of image patches and saves preliminary binary masks.

        :param
        ----------
        patches_rgb : List[Tuple[Tuple[int, int], np.ndarray]]
            A list of inputs, where each item is: ((row, col), image_rgb_array).

        num_rows : int
            Total number of rows in the grid (used for visualization).

        num_cols : int
            Total number of columns in the grid (used for visualization).

        save_dir : str
            Directory path where binary BMP masks will be saved.

        mask_ID : int
            Identifier for the current frame/mask (used for file naming).

        :return
        -------
        List[Tuple[Tuple[int, int], np.ndarray, List[dict]]]
            A list of results: ((row, col), original_patch, list_of_SAM_annotations).
            Each annotation is a dict containing keys like 'segmentation', 'area', 'bbox'.
        """


        results = []

        for (r, c), patch_rgb in patches_rgb:

            anns = self.mask_generator.generate(patch_rgb)

            results.append(((r, c), patch_rgb, anns))

        # visualize in real grid
        visualize_mask_grid_place_aware(
            results,
            num_rows=num_rows,
            num_cols=num_cols
        )

        # save binary immediately (after wiping folder)
        self.save_binary_masks(
            patches=results,
            save_dir=save_dir,
            mask_ID=mask_ID
        )

        return results   # RAW SAM OUTPUT

    def segment_refined(
        self,
        primary_results: List[Tuple[Tuple[int, int], np.ndarray, List[dict]]],
        num_rows: int,
        num_cols: int,
        processes: int = 4,
        **filter_kwargs
    ) -> List[Tuple[Tuple[int, int], np.ndarray, List[dict]]]:
        """
        Refines SAM segmentation masks using geometric heuristics and spatial constraints.

        This filter operates in two phases to remove artifacts while preserving "troublesome"
        larger areas if they are statistically significant.

        Phase 1: Geometric Pre-filtering (Shape Analysis)
        -------------------------------------------------
        Removes unstable shapes likely to be noise, cracks, or shadows.
        * **Contour Analysis:** Calculates the contour and convex hull of every mask.
        * **Solidity Check:** If a shape is highly elongated (high aspect ratio) AND
            non-convex (low solidity, i.e., much smaller than its hull), it is discarded.
            This effectively targets thin, squiggly artifacts.

        Phase 2: Vectorized Spatial Conflict (NMS-like Logic)
        -----------------------------------------------------
        Removes redundant masks based on overlap, specifically targeting fragmentation
        where a large object is split into many smaller sub-masks.

        * **Safety Guard:** Masks larger than `safe_size_threshold` are "anchors" and
            are NEVER removed, regardless of overlaps.
        * **Neighbor Selection:** For a mask `i`, we only look for conflict with neighbors `j`
            that satisfy:
            1. `Area(i) > Area(j)` (We only penalize the larger mask for having smaller debris inside).
            2. `Area(j)` is within `neighbor_size_range` relative to `Area(i)`.
        * **Overlap Check:** If mask `i` overlaps with more than `max_overlap_count`
            neighbors (based on `overlap_threshold`), mask `i` is discarded.

        :param
        ----------
        primary_results : List
            The output from `segment_primary`: list of ((row,col), patch_rgb, anns_raw).

        num_rows : int
            Total number of rows for visualization grid.

        num_cols : int
            Total number of cols for visualization grid.

        processes : int, optional
            Number of CPU processes to spawn for parallel filtering. Default is 4.

        **filter_kwargs : dict, optional
            Dynamic arguments passed to the heuristic filter. Supported keys:

            max_area_ratio: float
                Area above this threshold will be removed (possibly belonging to background region)

            aspect_threshold : float
                Max aspect ratio (long/short) allowed before solidity check is triggered.

            min_solidity : float
                Minimum solidity (contour_area / hull_area) required for elongated masks.

            overlap_threshold : float
                Intersection-Over-Area ratio required to count a neighbor as a "conflict".

            neighbor_size_range : Tuple[float, float]
                (min, max) ratio. Neighbor `j` considered if `low <= Area(j)/Area(i) <= high`.

            safe_size_threshold : float
                Masks larger than this fraction of image (0.0-1.0) are exempt from removal.

            max_overlap_count : int
                Max number of overlaps allowed before removal.

        :return
        -------
        List[Tuple[Tuple[int, int], np.ndarray, List[dict]]]
            Refined results list: ((row, col), patch_rgb, anns_filtered).
        """

        jobs = [
            ((r, c), patch_rgb, anns_raw, filter_kwargs)
            for (r, c), patch_rgb, anns_raw in primary_results
        ]

        if (processes > 1) and type(processes) == int:
            ctx = get_context("spawn")  # safest everywhere

            with ctx.Pool(processes=processes) as pool:
                refined_results = pool.map(refine_one_patch, jobs)

        else:
            refined_results = [refine_one_patch(job) for job in jobs]

        visualize_mask_grid_place_aware(
            refined_results,
            num_rows=num_rows,
            num_cols=num_cols
        )

        return refined_results

    @staticmethod
    def save_binary_masks(
            patches: List[Tuple[Tuple[int, int], np.ndarray, List[dict]]],
            save_dir: str,
            mask_ID: int,
    ) -> None:
        """
        Saves a single merged binary mask for every patch in the list.

        **WARNING:** This method clears the target directory before saving.

        :param
        ----------
        patches : List
            List of patch data: ((row,col), image_rgb, anns).

        save_dir : str
            Target directory to save BMP files.

        mask_ID : int
            Prefix for the output filenames (e.g., "{mask_ID}_{r}_{c}.bmp").

        :return
        -------
        None
        """

        os.makedirs(save_dir, exist_ok=True)

        # CLEAR DIRECTORY FIRST
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))

        for (r, c), _, anns in patches:

            if not anns:
                continue

            mask_canvas = np.zeros_like(anns[0]['segmentation'], dtype=np.uint8)

            for ann in anns:
                mask_canvas[ann['segmentation']] = 1

            fname = f"{mask_ID}_{r}_{c}.bmp"
            path = os.path.join(save_dir, fname)

            cv2.imwrite(path, (mask_canvas * 255).astype(np.uint8))


