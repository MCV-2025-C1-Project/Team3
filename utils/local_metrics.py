import cv2
import numpy as np
import math

RATIO_TEST = 0.75


def get_matcher(distance_type="L2", crossCheck=False):
    """
    Create a BFMatcher according to the distance type.
    Accepted values: "L2", "L1", "L2SQR", "HAMMING", "HAMMING2"
    """
    distance_type = distance_type.upper()
    norm_map = {
        "L1": cv2.NORM_L1,
        "L2": cv2.NORM_L2,
        "L2SQR": cv2.NORM_L2SQR,
        "HAMMING": cv2.NORM_HAMMING,
        "HAMMING2": cv2.NORM_HAMMING2
    }
    norm = norm_map.get(distance_type, cv2.NORM_L2)
    return cv2.BFMatcher(norm, crossCheck=crossCheck)


def _extract_descs(d):
    """
    Extract descriptors and keypoints arrays.
    Returns: (descs, kps)
    """
    descs = d.get("descriptors", None)
    kps = d.get("keypoints", None)
    if descs is None:
        return np.zeros((0, 128), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(descs, dtype=np.float32), np.asarray(kps, dtype=np.float32) if kps is not None else np.zeros((0, 4), dtype=np.float32)


# Basic matcher — no Lowe ratio
def sift_match_basic(dev_desc, db_desc, distance_type="L2"):
    """
    Basic matching: count of raw matches (no Lowe ratio).
    """
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    if des1.size == 0 or des2.size == 0:
        return 0.0

    BF = get_matcher(distance_type)
    matches = BF.match(des1, des2)
    return float(len(matches))


# Lowe ratio — raw count
def sift_match_count(dev_desc, db_desc, distance_type="L2"):
    """
    Count of good matches (Lowe ratio). Returns integer count (as float).
    """
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    if des1.size == 0 or des2.size == 0:
        return 0.0

    BF = get_matcher(distance_type)
    matches = BF.knnMatch(des1, des2, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good += 1
    return float(good)


# Normalized score
def sift_match_normalized(dev_desc, db_desc, distance_type="L2"):
    """
    Normalized score: good_matches / sqrt(N1 * N2)
    Useful when number of keypoints varies a lot.
    """
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    n1, n2 = des1.shape[0], des2.shape[0]
    if n1 == 0 or n2 == 0:
        return 0.0

    BF = get_matcher(distance_type)
    matches = BF.knnMatch(des1, des2, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good += 1

    denom = math.sqrt(n1 * n2)
    return float(good) / denom if denom > 0 else 0.0


# Geometric verification (RANSAC)
def sift_match_geometric(dev_desc, db_desc, distance_ftype="L2", reproj_thresh=5.0):
    """
    Lowe ratio + RANSAC-based geometric check.
    Returns number of inliers (higher = better). If fails returns 0.
    """
    des1, kp1 = _extract_descs(dev_desc)
    des2, kp2 = _extract_descs(db_desc)

    if des1.size == 0 or des2.size == 0 or kp1.size == 0 or kp2.size == 0:
        return 0.0

    BF = get_matcher(distance_type)
    matches = BF.knnMatch(des1, des2, k=2)
    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0

    src_pts = np.float32([kp1[m.queryIdx, :2] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx, :2] for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None or mask is None:
        return 0.0

    return float(np.sum(mask))
