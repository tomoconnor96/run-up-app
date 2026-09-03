"""
Automatic phase segmentation for a fast bowling delivery from a tracked
bounding-box signal. Heuristic, motion-based (no joint landmarks yet) --
see README / report caveats.
"""
import numpy as np
from scipy.signal import savgol_filter, find_peaks


def _smooth(y, win, poly=2):
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 5:
        return y.copy()
    if win % 2 == 0:
        win -= 1
    return savgol_filter(y, win, min(poly, win - 1))


def detect_phases(cx, cy, top, bottom, area, fps):
    n = len(bottom)

    # perspective drift baseline (bowler moves toward/away from camera)
    baseline = _smooth(bottom, win=max(9, n // 3))
    light = _smooth(bottom, win=5)
    residual = light - baseline  # negative = higher off the ground than trend

    # horizontal speed (run-up momentum) and overall motion magnitude
    vx = np.gradient(_smooth(cx, 7))
    vy = np.gradient(light)
    motion_mag = np.abs(vx) + np.abs(vy) + np.abs(np.gradient(_smooth(area, 7))) / (np.nanmax(area) + 1e-6)

    # --- gather apex: most-airborne point relative to local trend ---
    search_lo, search_hi = int(n * 0.10), int(n * 0.85)
    if search_hi <= search_lo:
        search_lo, search_hi = 0, n
    apex = search_lo + int(np.argmin(residual[search_lo:search_hi]))

    # --- back foot contact: first strong return toward baseline after apex ---
    bfc = None
    for i in range(apex + 1, min(apex + int(fps * 0.6), n)):
        if residual[i] >= residual[apex] * 0.35:  # recovered most of the way to ground
            bfc = i
            break
    if bfc is None:
        bfc = min(apex + max(2, int(fps * 0.12)), n - 1)

    # --- front foot contact: next landing after a short single-leg stride ---
    stride_guess = max(2, int(fps * 0.18))
    ffc_search_lo = bfc + 1
    ffc_search_hi = min(bfc + int(fps * 0.5), n - 1)
    ffc = None
    if ffc_search_hi > ffc_search_lo:
        # look for a secondary small dip+recover in residual after BFC
        seg = residual[ffc_search_lo:ffc_search_hi]
        if len(seg) > 2:
            peaks, _ = find_peaks(-seg)
            if len(peaks):
                ffc = ffc_search_lo + int(peaks[0])
    if ffc is None:
        ffc = min(bfc + stride_guess, n - 1)

    # --- release / snap: peak motion magnitude shortly after FFC ---
    rel_lo = ffc
    rel_hi = min(ffc + int(fps * 0.3), n)
    if rel_hi > rel_lo:
        release = rel_lo + int(np.argmax(motion_mag[rel_lo:rel_hi]))
    else:
        release = min(ffc + 1, n - 1)
    release = max(release, ffc)

    # --- gather start (end of run-up): takeoff point before apex ---
    gather_start = apex
    for i in range(apex, max(apex - int(fps * 0.5), 0), -1):
        if residual[i] >= -abs(residual[apex]) * 0.15:
            gather_start = i
            break

    run_up_end = max(gather_start - 1, 0)
    follow_through_end = n - 1

    phases = [
        ("RUN-UP", 0, run_up_end),
        ("GATHER", gather_start, max(gather_start, apex)),
        ("BACK FOOT CONTACT", apex + 1 if bfc <= apex else apex, bfc),
        ("STRIDE", bfc + 1, max(bfc + 1, ffc - 1)),
        ("FRONT FOOT CONTACT", ffc, ffc),
        ("RELEASE / SNAP", ffc + 1 if release <= ffc else release, release),
        ("FOLLOW-THROUGH", release + 1, follow_through_end),
    ]
    # clean overlaps / ordering
    cleaned = []
    prev_end = -1
    for name, a, b in phases:
        a = max(a, prev_end + 1)
        b = max(b, a)
        b = min(b, n - 1)
        cleaned.append((name, a, b))
        prev_end = b

    events = {
        "gather_apex": apex,
        "back_foot_contact": bfc,
        "front_foot_contact": ffc,
        "release": release,
    }
    signals = {
        "residual": residual,
        "motion_mag": motion_mag,
        "baseline": baseline,
    }
    return cleaned, events, signals


def score_phases(cleaned_phases, events, signals, cx, cy, top, bottom, area, fps):
    residual = signals["residual"]
    motion_mag = signals["motion_mag"]
    n = len(bottom)

    def clamp(v, lo=1, hi=10):
        return int(max(lo, min(hi, round(v))))

    scores = {}
    notes = {}

    apex = events["gather_apex"]
    jump_amt = abs(residual[apex])
    scale = np.nanmax(np.abs(residual)) + 1e-6
    gather_score = clamp(4 + 6 * (jump_amt / scale))
    scores["GATHER"] = gather_score
    notes["GATHER"] = "Clear, compact leap into the gather." if gather_score >= 7 else \
        "Some load into the gather, but it looks a little flat -- hard to be fully sure from this angle/tracking."

    bfc = events["back_foot_contact"]
    land_sharpness = abs(residual[min(bfc + 1, n - 1)] - residual[max(bfc - 1, 0)])
    bfc_score = clamp(4 + 6 * (land_sharpness / (scale + 1e-6)))
    scores["BACK FOOT CONTACT"] = bfc_score
    notes["BACK FOOT CONTACT"] = "Firm, positive landing." if bfc_score >= 7 else \
        "Landing detected, but the signal is soft -- treat this one as low-confidence."

    ffc = events["front_foot_contact"]
    stride_len = np.hypot(cx[ffc] - cx[bfc], cy[ffc] - cy[bfc])
    body_scale = np.nanmedian(bottom - top) + 1e-6
    stride_score = clamp(3 + 7 * (stride_len / (body_scale * 1.4)))
    scores["FRONT FOOT CONTACT"] = stride_score
    notes["FRONT FOOT CONTACT"] = "Reasonable stride into a braced-looking landing, but bracing angle itself " \
        "needs a side-on camera to actually verify -- take this score as a rough placeholder."

    release = events["release"]
    rel_lo, rel_hi = max(0, release - 2), min(n - 1, release + 2)
    rel_motion = np.nanmax(motion_mag[rel_lo:rel_hi + 1])
    motion_scale = np.nanmax(motion_mag) + 1e-6
    release_score = clamp(4 + 6 * (rel_motion / motion_scale))
    scores["RELEASE / SNAP"] = release_score
    notes["RELEASE / SNAP"] = "Sharp, high-energy release point." if release_score >= 7 else \
        "Release detected but without a strong motion spike -- worth a closer look on a slower, clearer clip."

    ft_start = release + 1
    if ft_start < n - 2:
        ft_motion = motion_mag[ft_start:]
        decay_smoothness = 10 - clamp(np.nanstd(ft_motion) / (np.nanmean(ft_motion) + 1e-6) * 4, 0, 9)
        ft_score = clamp(decay_smoothness)
    else:
        ft_score = 5
    scores["FOLLOW-THROUGH"] = ft_score
    notes["FOLLOW-THROUGH"] = "Long, smooth continuation through the crease." if ft_score >= 7 else \
        "Follow-through looks a bit abrupt or was cut short in this clip."

    scores["RUN-UP"] = None
    notes["RUN-UP"] = "Not scored automatically yet -- run-up rhythm scoring is on the roadmap " \
        "(needs the bowler in-frame for the full approach)."

    return scores, notes
