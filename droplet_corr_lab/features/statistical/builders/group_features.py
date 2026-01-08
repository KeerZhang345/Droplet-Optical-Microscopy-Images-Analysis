from typing import List, Dict, Callable

def extract_features_for_group(
    seg_rgb,
    group_coords,
    descriptors: List[Callable],
) -> Dict[str, float]:

    feats = {}
    for desc in descriptors:
        d = desc(seg_rgb, group_coords)
        feats.update(d)

    return feats

def extract_features_for_groups(
    seg_rgb,
    groups: List[List[List[int]]],
    descriptors: List[Callable],
    group_prefix: str = "g",
) -> List[Dict[str, float]]:
    """
    Returns list of dicts: one dict per group.
    """

    out = []
    for i, g in enumerate(groups):
        d = extract_features_for_group(seg_rgb, g, descriptors)
        d = {f"{group_prefix}_{i}_{k}": v for k, v in d.items()}
        out.append(d)

    return out

