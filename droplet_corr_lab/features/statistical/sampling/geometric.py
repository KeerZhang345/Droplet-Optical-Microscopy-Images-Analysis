import numpy as np

def grid_blocks(mask, block_size, stride):
    H, W = mask.shape
    groups = []

    for r in range(0, H - block_size + 1, stride):
        for c in range(0, W - block_size + 1, stride):
            block_mask = mask[r:r+block_size, c:c+block_size]
            if np.any(block_mask):
                rr, cc = np.where(block_mask)
                rr += r
                cc += c
                groups.append(list(zip(rr.tolist(), cc.tolist())))

    return groups


def grid_radial_slices(mask, n_slices):
    H, W = mask.shape
    cy, cx = H // 2, W // 2

    yy, xx = np.indices(mask.shape)
    angles = (np.arctan2(yy - cy, xx - cx) + 2*np.pi) % (2*np.pi)

    bins = np.linspace(0, 2*np.pi, n_slices + 1)
    groups = []

    for i in range(n_slices):
        m = (angles >= bins[i]) & (angles < bins[i+1]) & mask
        if np.any(m):
            groups.append(list(zip(*np.where(m))))

    return groups


def grid_ring_slices(mask, n_rings):
    H, W = mask.shape
    cy, cx = H // 2, W // 2

    yy, xx = np.indices(mask.shape)
    radii = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    rmin, rmax = radii[mask].min(), radii[mask].max()
    bins = np.linspace(rmin, rmax, n_rings + 1)

    groups = []

    for i in range(n_rings):
        m = (radii >= bins[i]) & (radii < bins[i+1]) & mask
        if np.any(m):
            groups.append(list(zip(*np.where(m))))

    return groups
