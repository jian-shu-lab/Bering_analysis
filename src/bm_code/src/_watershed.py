import cv2
import logging
from skimage import filters

from typing import Optional, Tuple
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature, filters, measure, segmentation

from . import plotting as pl
logger = logging.getLogger(__name__)

### refer to segmentation procedure in spateo
def mclose_mopen(mask: np.ndarray, k: int, square: bool = False) -> np.ndarray:
    """Perform morphological close and open operations on a boolean mask.
    Args:
        X: Boolean mask
        k: Kernel size
        square: Whether or not the kernel should be square
    Returns:
        New boolean mask with morphological close and open operations performed.
    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    kernel = np.ones((k, k), dtype=np.uint8) # if square else circle(k)
    mclose = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mopen = cv2.morphologyEx(mclose, cv2.MORPH_OPEN, kernel)

    return mopen.astype(bool)

def _mask_nuclei_from_stain(
    X: np.ndarray,
    otsu_classes: int = 3,
    otsu_index: int = 0,
    local_k: int = 55,
    offset: int = 5,
    mk: int = 5,
) -> np.ndarray:
    """Create a boolean mask indicating nuclei from stained nuclei image.
    See :func:`mask_nuclei_from_stain` for arguments.
    """
    thresholds = filters.threshold_multiotsu(X, otsu_classes)
    background_mask = X < thresholds[otsu_index]

    local_mask = cv2.adaptiveThreshold(
        X, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, local_k, offset
    ).astype(bool)

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    nuclei_mask = mclose_mopen((~background_mask) & local_mask, mk)
    return nuclei_mask

def _find_peaks(X: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks from an arbitrary image.
    This function is a wrapper around :func:`feature.peak_local_max`.
    Args:
        X: Array to find peaks from
        **kwargs: Keyword arguments to pass to :func:`feature.peak_local_max`.
    Returns:
        Numpy array of the same size as `X` where each peak is labeled with a unique positive
        integer.
    """
    _kwargs = dict(p_norm=2)
    _kwargs.update(kwargs)
    peak_idx = feature.peak_local_max(X, **_kwargs)
    peaks = np.zeros(X.shape, dtype=int)
    for label, (i, j) in enumerate(peak_idx):
        peaks[i, j] = label + 1
    return peaks

def find_peaks_from_mask(
    mask,
    min_distance: int,
    mask_size: int = 3,
):
    """Find peaks from a boolean mask. Used to obatin Watershed markers.
    Args:
        adata: Input AnnData
        layer: Layer containing boolean mask. This will default to `{layer}_mask`.
            If not present in the provided AnnData, this argument used as a literal.
        min_distance: Minimum distance, in pixels, between peaks.
        distances_layer: Layer to save distance from each pixel to the nearest zero (False)
            pixel (a.k.a. distance transform). By default, uses `{layer}_distances`.
        markers_layer: Layer to save identified peaks as markers. By default, uses
            `{layer}_markers`.
    """
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, mask_size)
    peaks = _find_peaks(distances, min_distance=min_distance)
    return peaks

def _watershed(
    X: np.ndarray,
    mask: np.ndarray,
    markers: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Assign individual nuclei/cells using the Watershed algorithm.
    Args:
        X: Data array. This array will be Gaussian blurred and used as the
            input values to Watershed.
        mask: Nucleus/cell mask.
        markers: Numpy array indicating where the Watershed markers are. May
            either be a boolean or integer array. If this is a boolean array,
            the markers are identified by calling `cv2.connectedComponents`.
        k: Size of the kernel to use for Gaussian blur.
    Returns:
        Watershed labels.
    """
    # blur = utils.conv2d(X, k, mode="gauss")
    blur = cv2.GaussianBlur(src=X.astype(float), ksize=(k, k), sigmaX=0, sigmaY=0)
    markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
    watershed = segmentation.watershed(-blur, markers, mask=mask)
    return watershed

def run_watershed(
    img, plot_name = None,
    multi_ostu_classes = 3,
    adaptive_thresh_blockSize = 55,
    mclose_mopen_mk = 5,
    dist_transform_ksize = 3,
    peak_min_distance = 5,
    gaussian_blur_ksize = 3,
):
    mask = _mask_nuclei_from_stain(img, otsu_classes = multi_ostu_classes, local_k = adaptive_thresh_blockSize, mk = mclose_mopen_mk)
    markers = find_peaks_from_mask(mask, min_distance = peak_min_distance, mask_size = dist_transform_ksize)
    watershed_res = _watershed(X = img, mask = mask, markers = markers, k = gaussian_blur_ksize)

    if plot_name is not None:
        pl.plot_watershed(img, watershed_res, output_name = plot_name)
    return watershed_res