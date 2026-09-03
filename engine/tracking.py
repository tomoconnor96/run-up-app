"""
Generalized bowler silhouette tracking for arbitrary uploaded clips.

Approach: background subtraction (MOG2) + nearest-neighbour contour
tracking, initialised on the largest early motion blob and then locked
on by proximity frame-to-frame (so it doesn't jump to unrelated people
moving elsewhere in the frame, e.g. a keeper or batter).
"""
import cv2
import numpy as np


def read_frames(video_path, max_dim=640):
    """Read frames, downscaling large videos so processing stays cheap
    on low-memory hosts (free-tier servers commonly have ~512MB RAM)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    scale = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale is None:
            h, w = frame.shape[:2]
            longest = max(h, w)
            scale = min(1.0, max_dim / longest)
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames, fps


def track_subject(frames, min_area=None, max_jump_frac=0.35):
    """Returns a list of dicts per frame: {bbox:(x,y,w,h) or None, area}."""
    H, W = frames[0].shape[:2]
    max_jump = max_jump_frac * ((W + H) / 2)
    if min_area is None:
        min_area = 0.0015 * W * H  # resolution-independent (frames may be downscaled)

    # Downscaled frames are cheap enough to afford two full warm-up passes,
    # which noticeably improves the background model / tracking quality.
    backSub = cv2.createBackgroundSubtractorMOG2(history=min(500, 2 * len(frames)), varThreshold=40, detectShadows=True)
    for _ in range(2):
        for f in frames:
            backSub.apply(f, learningRate=0.02)

    kernel = np.ones((5, 5), np.uint8)
    raw = []
    for f in frames:
        fg = backSub.apply(f, learningRate=0.0)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg[:, int(W * 0.97):] = 0
        fg[:, :int(W * 0.02)] = 0
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                cands.append({"bbox": (x, y, w, h), "area": area, "centroid": (x + w / 2, y + h / 2)})
        raw.append(cands)

    # seed: pick the frame in the first half with the single largest candidate
    seed_idx, seed_cand = None, None
    for i in range(len(raw) // 2):
        for c in raw[i]:
            if seed_cand is None or c["area"] > seed_cand["area"]:
                seed_idx, seed_cand = i, c
    if seed_cand is None:
        for i, cands in enumerate(raw):
            for c in cands:
                if seed_cand is None or c["area"] > seed_cand["area"]:
                    seed_idx, seed_cand = i, c

    track = [None] * len(frames)
    if seed_cand is None:
        return track

    track[seed_idx] = seed_cand
    last = seed_cand
    for i in range(seed_idx + 1, len(frames)):
        best, best_d = None, None
        for c in raw[i]:
            d = np.hypot(c["centroid"][0] - last["centroid"][0], c["centroid"][1] - last["centroid"][1])
            if d <= max_jump and (best_d is None or d < best_d):
                best, best_d = c, d
        if best:
            track[i] = best
            last = best
    last = seed_cand
    for i in range(seed_idx - 1, -1, -1):
        best, best_d = None, None
        for c in raw[i]:
            d = np.hypot(c["centroid"][0] - last["centroid"][0], c["centroid"][1] - last["centroid"][1])
            if d <= max_jump and (best_d is None or d < best_d):
                best, best_d = c, d
        if best:
            track[i] = best
            last = best

    return track


def interpolate_track(track, n):
    """Fill gaps in centroid/bbox by linear interpolation; returns arrays."""
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    top = np.full(n, np.nan)
    bottom = np.full(n, np.nan)
    area = np.full(n, np.nan)
    for i, t in enumerate(track):
        if t is not None:
            x, y, w, h = t["bbox"]
            cx[i], cy[i] = x + w / 2, y + h / 2
            top[i], bottom[i] = y, y + h
            area[i] = t["area"]

    def fill(arr):
        idx = np.arange(n)
        good = ~np.isnan(arr)
        if good.sum() < 2:
            return np.nan_to_num(arr)
        arr[~good] = np.interp(idx[~good], idx[good], arr[good])
        return arr

    return fill(cx), fill(cy), fill(top), fill(bottom), fill(area)
