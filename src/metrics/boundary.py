"""Boundary-aware metrics: HD95 and boundary IoU. Clinically meaningful for polyp
margins. Operate on single HxW binary numpy masks (per-sample, not batched).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance between predicted and ground-truth
    contours (Euclidean distance transform based). Returns NaN if either mask is empty.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")

    pred_boundary = pred & ~ndimage.binary_erosion(pred)
    gt_boundary = gt & ~ndimage.binary_erosion(gt)

    gt_dist = ndimage.distance_transform_edt(~gt)
    pred_dist = ndimage.distance_transform_edt(~pred)

    dists = np.concatenate([gt_dist[pred_boundary], pred_dist[gt_boundary]])
    return float(np.percentile(dists, 95))


def boundary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, dilation: int = 2) -> float:
    """IoU restricted to a narrow band around the mask boundary (Cheng et al. 2021):
    `Boundary(mask, d) = mask \\ erode(mask, d)`. Returns NaN if both bands are empty.
    """
    pred_band = _boundary_band(pred_mask.astype(bool), dilation)
    gt_band = _boundary_band(gt_mask.astype(bool), dilation)

    union = np.logical_or(pred_band, gt_band).sum()
    if union == 0:
        return float("nan")
    intersection = np.logical_and(pred_band, gt_band).sum()
    return float(intersection / union)


def _boundary_band(mask: np.ndarray, dilation: int) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=dilation)
    return mask & ~eroded
