"""
=============================================================================
sort_tracker.py  —  Simple Online and Realtime Tracking (SORT)
Project: Vehicle & Pedestrian Tracking (CO5430 – Group G07)
=============================================================================

This is a self-contained implementation of the SORT tracker algorithm.

SORT paper: "Simple Online and Realtime Tracking" (Bewley et al., 2016)
https://arxiv.org/abs/1602.00763

HOW SORT WORKS (short version):
  Every frame:
    1. YOLOv8 gives us a list of bounding boxes (detections)
    2. SORT has a list of active tracks (objects from previous frames)
    3. Kalman Filter predicts where each track SHOULD be now
    4. Hungarian Algorithm matches predictions to new detections
    5. Matched → update track, keep same ID
       Unmatched detection → create new track
       Unmatched track too long → delete it

DEPENDENCIES:
  pip install filterpy scipy numpy

=============================================================================
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# =============================================================================
# KALMAN FILTER TRACKER  (one per tracked object)
# =============================================================================
class KalmanBoxTracker:
    """
    A single object tracker using a Kalman Filter.

    State vector: [cx, cy, aspect_ratio, height, dx, dy, da, dh]
      cx, cy   = center x, y (in pixels)
      ar       = aspect ratio (width/height)
      h        = height (in pixels)
      dx, dy, da, dh = velocities

    Observation: [cx, cy, ar, h]  (from bounding box detection)
    """

    count = 0   # class-level counter for assigning unique IDs

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]  in pixel coordinates
        """
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # ── State transition matrix (constant-velocity model) ──
        # x[t+1] = F · x[t]
        self.kf.F = np.array([
            [1,0,0,0, 1,0,0,0],   # cx  += dx
            [0,1,0,0, 0,1,0,0],   # cy  += dy
            [0,0,1,0, 0,0,1,0],   # ar  += da
            [0,0,0,1, 0,0,0,1],   # h   += dh
            [0,0,0,0, 1,0,0,0],   # dx stays
            [0,0,0,0, 0,1,0,0],   # dy stays
            [0,0,0,0, 0,0,1,0],   # da stays
            [0,0,0,0, 0,0,0,1],   # dh stays
        ], dtype=float)

        # ── Measurement matrix: we observe only positions (not velocities) ──
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)

        # ── Noise matrices ──
        self.kf.R[2:, 2:] *= 10.   # measurement noise (aspect ratio & height less certain)
        self.kf.P[4:, 4:] *= 1000. # initial velocity uncertainty (high — we don't know initial speed)
        self.kf.P         *= 10.
        self.kf.Q[-1, -1] *= 0.01  # process noise (height change)
        self.kf.Q[4:, 4:] *= 0.01  # process noise (velocities)

        # Initialise state from first detection
        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0  # frames since last matched detection
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0          # total successful matches
        self.hit_streak = 0    # consecutive hits (used for min_hits check)
        self.age = 0           # total frames this track has existed

    @classmethod
    def reset_count(cls):
        """Reset the global ID counter (call when starting a new sequence)."""
        cls.count = 0

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_z(bbox):
        """Convert [x1,y1,x2,y2] → [cx, cy, ar, h] (column vector)."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        ar = w / float(h) if h > 0 else 1.0
        return np.array([[cx], [cy], [ar], [h]], dtype=float)

    @staticmethod
    def _z_to_bbox(x, score=None):
        """Convert Kalman state [cx,cy,ar,h,...] → [x1,y1,x2,y2] (optionally with score)."""
        w = x[2] * x[3]   # ar * h = width
        x1 = x[0] - w / 2.0
        y1 = x[1] - x[3] / 2.0
        x2 = x[0] + w / 2.0
        y2 = x[1] + x[3] / 2.0
        if score is None:
            return np.array([x1, y1, x2, y2]).flatten()
        return np.array([x1, y1, x2, y2, score]).flatten()

    # ── Main methods ───────────────────────────────────────────────────────

    def update(self, bbox):
        """Feed a new matched detection into the Kalman filter."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self):
        """Advance the state estimate by one time step (one frame)."""
        # Prevent negative width (ar*h must be positive)
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return current bounding box estimate [x1,y1,x2,y2]."""
        return self._z_to_bbox(self.kf.x)


# =============================================================================
# IoU COMPUTATION  (used by the Hungarian matcher)
# =============================================================================
def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.

    bb_test : (N, 4)  [x1,y1,x2,y2]
    bb_gt   : (M, 4)  [x1,y1,x2,y2]
    Returns : (N, M)  IoU values
    """
    bb_gt = np.expand_dims(bb_gt, 0)   # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = ((bb_test[..., 2] - bb_test[..., 0]) *
                 (bb_test[..., 3] - bb_test[..., 1]))
    area_gt   = ((bb_gt[..., 2]   - bb_gt[..., 0])   *
                 (bb_gt[..., 3]   - bb_gt[..., 1]))

    iou = inter / (area_test + area_gt - inter + 1e-9)
    return iou   # shape: (N, M)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Match detections to existing tracks using Hungarian algorithm.

    detections : (N, 5) [x1,y1,x2,y2,score]
    trackers   : (M, 4) [x1,y1,x2,y2]

    Returns:
      matches       : (K, 2)  matched [det_idx, trk_idx] pairs
      unmatched_det : unmatched detection indices
      unmatched_trk : unmatched tracker indices
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    iou_matrix = iou_batch(detections[:, :4], trackers)  # (N, M)

    # Hungarian: minimise cost = maximise IoU
    # linear_sum_assignment minimises, so negate IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    # Filter out matches with IoU below threshold
    matches, unmatched_det, unmatched_trk = [], [], []

    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_det.append(d)

    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trk.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_det.append(m[0])
            unmatched_trk.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = (np.concatenate(matches, axis=0)
               if len(matches) > 0
               else np.empty((0, 2), dtype=int))

    return matches, np.array(unmatched_det), np.array(unmatched_trk)


# =============================================================================
# SORT  —  The main tracker class
# =============================================================================
class Sort:
    """
    SORT multi-object tracker.

    Parameters
    ----------
    max_age       : int   How many frames a track survives without a detection.
                          (Default 1 = delete immediately if not matched)
    min_hits      : int   Minimum detections before a new track is reported.
                          Avoids reporting false alarms.
    iou_threshold : float Minimum IoU to consider a detection-track match.
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.reset_count()  # reset IDs so each sequence starts from 0

    def update(self, dets=np.empty((0, 5))):
        """
        Update SORT with new detections.

        Parameters
        ----------
        dets : np.ndarray  shape (N, 5)
               Each row: [x1, y1, x2, y2, confidence]
               If no detections: pass np.empty((0,5))

        Returns
        -------
        np.ndarray  shape (M, 5)
               Each row: [x1, y1, x2, y2, track_id]
               Only returns tracks that have been confirmed (≥ min_hits).
        """
        self.frame_count += 1

        # ── Step 1: Predict new positions for all existing tracks ──
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove degenerate trackers (iterate in reverse to keep indices valid)
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Rebuild trks cleanly from the surviving trackers so indices are correct
        trks = (np.array([trk.get_state() for trk in self.trackers])
                if self.trackers else np.empty((0, 4)))

        # ── Step 2: Match detections to predictions ──
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # ── Step 3: Update matched trackers ──
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # ── Step 4: Create new tracks for unmatched detections ──
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # ── Step 5: Remove stale tracks ──
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # Only return confirmed tracks (enough hits, recently seen)
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        return (np.concatenate(ret) if ret else np.empty((0, 5)))
