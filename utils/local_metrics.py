import cv2
import numpy as np
import math

# BF matcher for SIFT (L2)
BF = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
RATIO_TEST = 0.75

def _extract_descs(d):
    """
    Accepts the dict or tuple used in your pipeline and returns descriptors (Nx128) and keypoints array (Nx4)
    """
    descs = d.get("descriptors", None)
    kps = d.get("keypoints", None)
    if descs is None:
        return np.zeros((0, 128), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(descs, dtype=np.float32), np.asarray(kps, dtype=np.float32) if kps is not None else np.zeros((0, 4), dtype=np.float32)


def sift_match_count(dev_desc, db_desc):
    """
    Count of good matches (Lowe ratio). Returns integer count (as float).
    """
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    if des1.size == 0 or des2.size == 0:
        return 0.0

    matches = BF.knnMatch(des1, des2, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good += 1
    return float(good)


def sift_match_normalized(dev_desc, db_desc):
    """
    Normalized score: good_matches / sqrt(N1 * N2)
    Useful when number of keypoints varies a lot.
    """
    des1, _ = _extract_descs(dev_desc)
    des2, _ = _extract_descs(db_desc)

    n1 = des1.shape[0]
    n2 = des2.shape[0]
    if n1 == 0 or n2 == 0:
        return 0.0

    matches = BF.knnMatch(des1, des2, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good += 1

    denom = math.sqrt(n1 * n2)
    if denom == 0:
        return 0.0
    return float(good) / denom


def sift_match_geometric(dev_desc, db_desc, reproj_thresh=5.0):
    """
    Lowe ratio -> compute homography via RANSAC using matched keypoint coordinates
    Returns number of inliers (higher = better). If homography fails returns 0.
    dev_desc and db_desc must contain 'keypoints' arrays (Nx4).
    """
    des1, kp1 = _extract_descs(dev_desc)
    des2, kp2 = _extract_descs(db_desc)

    if des1.size == 0 or des2.size == 0 or kp1.size == 0 or kp2.size == 0:
        return 0.0

    # knn matching
    matches = BF.knnMatch(des1, des2, k=2)
    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0  # not enough matches for homography

    # build src/dst point arrays for cv2.findHomography
    src_pts = []
    dst_pts = []
    for m in good_matches:
        # m.queryIdx -> index in des1 / kp1 (dev)
        # m.trainIdx -> index in des2 / kp2 (db)
        q = int(m.queryIdx)
        t = int(m.trainIdx)
        # kp arrays are (x, y, size, angle)
        src_pts.append([kp1[q, 0], kp1[q, 1]])
        dst_pts.append([kp2[t, 0], kp2[t, 1]])

    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None or mask is None:
        return 0.0

    inliers = int(np.sum(mask))
    return float(inliers)
