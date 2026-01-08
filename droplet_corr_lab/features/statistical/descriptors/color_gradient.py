import numpy as np
import cv2
from typing import List, Dict

def _stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan}

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        }

def rgb_features(
    image_rgb: np.ndarray,
    coords: List[List[int]],
) -> Dict[str, float]:

    coords = np.asarray(coords)
    pixels = image_rgb[coords[:, 0], coords[:, 1]]  # (N, 3)

    feats = {}
    for i, ch in enumerate(["R", "G", "B"]):
        s = _stats(pixels[:, i])
        for k, v in s.items():
            feats[f"{ch}_{k}"] = v

    return feats

def rgb_features_only_mean(
    image_rgb: np.ndarray,
    coords: List[List[int]],
) -> Dict[str, float]:

    coords = np.asarray(coords)
    pixels = image_rgb[coords[:, 0], coords[:, 1]]  # (N, 3)

    feats = {}
    for i, ch in enumerate(["R", "G", "B"]):
        s = _stats(pixels[:, i])
        feats[f"{ch}_mean"] = s["mean"]

    return feats

def hsv_features(
    image_rgb: np.ndarray,
    coords: List[List[int]],
) -> Dict[str, float]:

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    coords = np.asarray(coords)
    pixels = hsv[coords[:, 0], coords[:, 1]]

    feats = {}
    for i, ch in enumerate(["H", "S", "V"]):
        s = _stats(pixels[:, i])
        for k, v in s.items():
            feats[f"{ch}_{k}"] = v

    return feats

def hsv_features_only_mean(
    image_rgb: np.ndarray,
    coords: List[List[int]],
) -> Dict[str, float]:

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    coords = np.asarray(coords)
    pixels = hsv[coords[:, 0], coords[:, 1]]

    feats = {}
    for i, ch in enumerate(["H", "S", "V"]):
        s = _stats(pixels[:, i])
        feats[f"{ch}_mean"] = s["mean"]

    return feats

def pairwise_difference_features(pixel_values):
    # Compute all pairwise differences
    pixel_values = np.array(pixel_values, dtype=np.float64)
    differences = np.abs(pixel_values[:, None] - pixel_values).flatten()

    # Extract statistics
    feats = {
        "Gra_mean": np.mean(differences),
        "Gra_var": np.var(differences),
    }

    return feats


def pairwise_difference_features_sampled(pixel_values, max_samples=20000):
    """
    Compute stats of pairwise differences without O(N^2) memory blowup.
    If too many pixels, subsample random pairs.
    """
    pixel_values = np.asarray(pixel_values, dtype=np.float32).ravel()
    n = len(pixel_values)

    # Small regions — compute exactly
    if n < 1500:
        diffs = np.abs(pixel_values[:, None] - pixel_values).ravel()
        return {
            "Gra_mean": float(np.mean(diffs)),
            "Gra_var": float(np.var(diffs)),
        }

    # Large regions — random sampling
    rng = np.random.default_rng(42)
    i = rng.integers(0, n, size=max_samples)
    j = rng.integers(0, n, size=max_samples)

    diffs = np.abs(pixel_values[i] - pixel_values[j])

    return {
        "Gra_mean": float(np.mean(diffs)),
        "Gra_var": float(np.var(diffs)),
    }


def histogram_features(pixel_values, bins=5):
    hist, _ = np.histogram(pixel_values, bins=bins, range=(0, 255), density=True)
    feats = {}
    for i, val in enumerate(hist):
        feats[f"Gra_his_{i}"] = val

    return feats


def gradient_features(
    image_rgb: np.ndarray,
    coords: List[List[int]],
) -> Dict[str, float]:

    coords = np.asarray(coords)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pixels = gray[coords[:, 0], coords[:, 1]]  # (N, )

    feats = {}
    feats.update(pairwise_difference_features_sampled(pixels))
    feats.update(histogram_features(pixels))

    return feats