# utils/local_metrics.py
import cv2
import numpy as np
import math


def _extract_descs(d):
    """
    Extract descriptor and keypoint arrays WITHOUT forcing dtype for descriptors.
    Returns: (descs, kps)
      - descs: np.ndarray (N, D), dtype can be float32 (SIFT/SURF) or uint8 (ORB/BRISK/AKAZE)
      - kps:   np.ndarray (N, 2) with [x, y] or (0, 2) if not present
    """
    descs = d.get("descriptors", None)
    kps = d.get("keypoints", None)

    if descs is None or len(descs) == 0:
        return np.zeros((0, 128)), np.zeros((0, 2), dtype=np.float32)

    # Keep original dtype for descriptors (very important!)
    descs = np.asarray(descs)

    # Keypoints as float32 [x, y] (trim if more columns exist)
    if kps is None or len(kps) == 0:
        kps = np.zeros((0, 2), dtype=np.float32)
    else:
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape[1] >= 2:
            kps = kps[:, :2]
        else:
            tmp = np.zeros((kps.shape[0], 2), dtype=np.float32)
            tmp[:, :kps.shape[1]] = kps
            kps = tmp
    return descs, kps


def _infer_norm_from_dtype(des):
    """
    Choose the correct OpenCV norm based on descriptor dtype.
      - uint8  -> HAMMING
      - others -> L2
    """
    return cv2.NORM_HAMMING if des.dtype == np.uint8 else cv2.NORM_L2


def _norm_from_string(distance_type):
    """
    Map a string name to an OpenCV norm. Defaults to L2.
    """
    distance_type = (distance_type or "L2").upper()
    mapping = {
        "L1": cv2.NORM_L1,
        "L2": cv2.NORM_L2,
        "HAMMING": cv2.NORM_HAMMING,
    }
    return mapping.get(distance_type, cv2.NORM_L2), distance_type


def _get_safe_matcher(des1, des2, distance_type=None, for_knn=True, cross_check=False):
    """
    Build a BFMatcher with a norm that is compatible with descriptor dtype.
    - If the requested distance_type conflicts with dtype, prefer dtype-inferred norm.
    - crossCheck must be False when using knnMatch (ratio test), True only for match().
    """
    wanted_norm, _ = _norm_from_string(distance_type)
    inferred_norm = _infer_norm_from_dtype(des1)

    # Prefer inferred norm from dtype to avoid OpenCV assertion failures.
    norm = inferred_norm

    # If the requested norm is compatible with inferred, accept it (e.g., HAMMING2 vs HAMMING, L2 vs L1)
    if wanted_norm in (cv2.NORM_HAMMING, cv2.NORM_HAMMING2) and inferred_norm == cv2.NORM_HAMMING:
        norm = wanted_norm
    elif wanted_norm in (cv2.NORM_L2, cv2.NORM_L2SQR, cv2.NORM_L1) and inferred_norm == cv2.NORM_L2:
        norm = wanted_norm
    cross_check = True if not for_knn and cross_check else False
    return cv2.BFMatcher(norm, crossCheck=cross_check)


def _knn_with_fallback(des1, des2, k=2, distance_type=None):
    """
    Run knnMatch with a safe matcher. If OpenCV fails, retry with dtype-inferred norm.
    """
    BF = _get_safe_matcher(des1, des2, distance_type=distance_type, for_knn=True)
    try:
        return BF.knnMatch(des1, des2, k=k)
    except cv2.error:
        BF = cv2.BFMatcher(_infer_norm_from_dtype(des1), crossCheck=False)
        return BF.knnMatch(des1, des2, k=k)



# ----------------------------- Local matching metrics -----------------------------

def match_count(dev_desc, db_desc, distance_type="L2",ratio_test=0.75):
    """
    Raw count of 'good matches' after Lowe's ratio test (k=2).
    Returns float for pipeline consistency.
    """
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    if des1.dtype != des2.dtype:
        if des1.dtype == np.uint8 or des2.dtype == np.uint8:
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)

    if des1.size == 0 or des2.size == 0:
        return 0.0

    matches = _knn_with_fallback(des1, des2, k=2, distance_type=distance_type)

    good = 0
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good += 1
    return float(good)


def match_geometric(dev_desc, db_desc, distance_type="L2", reproj_thresh=5.0):
    des1, kp1 = _extract_descs(dev_desc)
    des2, kp2 = _extract_descs(db_desc)

    if des1.size == 0 or des2.size == 0 or kp1.shape[0] == 0 or kp2.shape[0] == 0:
        return 0.0

    if des1.dtype != des2.dtype:
        if des1.dtype == np.uint8 or des2.dtype == np.uint8:
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)
    BF = _get_safe_matcher(des1, des2, distance_type=distance_type, for_knn=True)
    try:
        matches = BF.knnMatch(des1, des2, k=2)
    except cv2.error:
        BF = cv2.BFMatcher(_infer_norm_from_dtype(des1), crossCheck=False)
        matches = BF.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0

    src_pts = np.float32([kp1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None or mask is None:
        return 0.0

    return float(np.sum(mask))



def match_reciprocal_count(dev_desc, db_desc, distance_type="L2"):
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    if des1.dtype != des2.dtype:
        if des1.dtype == np.uint8 or des2.dtype == np.uint8:
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)
        else:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)

    if des1.size == 0 or des2.size == 0:
        return 0.0

    try:
        BF = _get_safe_matcher(des1, des2, distance_type=distance_type, for_knn=False, cross_check=True)
        matches = BF.match(des1, des2)
    except cv2.error:
        BF = cv2.BFMatcher(_infer_norm_from_dtype(des1), crossCheck=True)
        matches = BF.match(des1, des2)

    return float(len(matches)) if matches else 0.0