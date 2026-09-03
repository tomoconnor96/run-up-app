"""
Phase segmentation and scoring from real joint landmarks (MediaPipe Pose).
"""
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from engine.pose import angle_deg


def _smooth(y, win, poly=2):
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 5:
        return y.copy()
    if win % 2 == 0:
        win -= 1
    return savgol_filter(y, win, min(poly, win - 1))


def _bowling_arm(joints, n):
    l_sh_y = joints["l_shoulder"][1]
    r_sh_y = joints["r_shoulder"][1]
    l_wr_y = joints["l_wrist"][1]
    r_wr_y = joints["r_wrist"][1]
    half = n // 2
    l_lift = np.max(l_sh_y[half:] - l_wr_y[half:]) if half < n else 0
    r_lift = np.max(r_sh_y[half:] - r_wr_y[half:]) if half < n else 0
    return "l" if l_lift >= r_lift else "r"


def detect_phases(joints, fps):
    n = len(joints["l_ankle"][0])
    ankle_mid_y = (joints["l_ankle"][1] + joints["r_ankle"][1]) / 2
    hip_mid_x = (joints["l_hip"][0] + joints["r_hip"][0]) / 2

    baseline = _smooth(ankle_mid_y, win=max(9, n // 3))
    light = _smooth(ankle_mid_y, win=5)
    residual = light - baseline

    search_lo, search_hi = int(n * 0.10), int(n * 0.85)
    if search_hi <= search_lo:
        search_lo, search_hi = 0, n
    apex = search_lo + int(np.argmin(residual[search_lo:search_hi]))

    bfc = None
    for i in range(apex + 1, min(apex + int(fps * 0.6), n)):
        if residual[i] >= residual[apex] * 0.35:
            bfc = i
            break
    if bfc is None:
        bfc = min(apex + max(2, int(fps * 0.12)), n - 1)

    ffc = None
    ffc_lo, ffc_hi = bfc + 1, min(bfc + int(fps * 0.5), n - 1)
    if ffc_hi > ffc_lo:
        seg = residual[ffc_lo:ffc_hi]
        if len(seg) > 2:
            peaks, _ = find_peaks(-seg)
            if len(peaks):
                ffc = ffc_lo + int(peaks[0])
    if ffc is None:
        ffc = min(bfc + max(2, int(fps * 0.18)), n - 1)

    bow_arm = _bowling_arm(joints, n)
    wrist_y = joints[f"{bow_arm}_wrist"][1]
    rel_lo, rel_hi = ffc, min(ffc + int(fps * 0.3), n)
    if rel_hi > rel_lo:
        release = rel_lo + int(np.argmin(wrist_y[rel_lo:rel_hi]))
    else:
        release = min(ffc + 1, n - 1)
    release = max(release, ffc)

    gather_start = apex
    for i in range(apex, max(apex - int(fps * 0.5), 0), -1):
        if residual[i] >= -abs(residual[apex]) * 0.15:
            gather_start = i
            break

    phases = [
        ("RUN-UP", 0, max(gather_start - 1, 0)),
        ("GATHER", gather_start, max(gather_start, apex)),
        ("BACK FOOT CONTACT", apex + 1 if bfc <= apex else apex, bfc),
        ("STRIDE", bfc + 1, max(bfc + 1, ffc - 1)),
        ("FRONT FOOT CONTACT", ffc, ffc),
        ("RELEASE / SNAP", ffc + 1 if release <= ffc else release, release),
        ("FOLLOW-THROUGH", release + 1, n - 1),
    ]
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
        "bowling_arm": bow_arm,
    }
    return cleaned, events


def _knee_angle(joints, side, i):
    hip = (joints[f"{side}_hip"][0][i], joints[f"{side}_hip"][1][i])
    knee = (joints[f"{side}_knee"][0][i], joints[f"{side}_knee"][1][i])
    ankle = (joints[f"{side}_ankle"][0][i], joints[f"{side}_ankle"][1][i])
    return float(angle_deg(hip, knee, ankle))


def _elbow_angle(joints, side, i):
    sh = (joints[f"{side}_shoulder"][0][i], joints[f"{side}_shoulder"][1][i])
    el = (joints[f"{side}_elbow"][0][i], joints[f"{side}_elbow"][1][i])
    wr = (joints[f"{side}_wrist"][0][i], joints[f"{side}_wrist"][1][i])
    return float(angle_deg(sh, el, wr))


def _torso_len(joints, i):
    shx = (joints["l_shoulder"][0][i] + joints["r_shoulder"][0][i]) / 2
    shy = (joints["l_shoulder"][1][i] + joints["r_shoulder"][1][i]) / 2
    hpx = (joints["l_hip"][0][i] + joints["r_hip"][0][i]) / 2
    hpy = (joints["l_hip"][1][i] + joints["r_hip"][1][i]) / 2
    return float(np.hypot(shx - hpx, shy - hpy)) + 1e-6


def _band_score(value, lo, hi, soft=15):
    if lo <= value <= hi:
        return 10
    d = (lo - value) if value < lo else (value - hi)
    return max(1, round(10 - d / soft * 6))


def score_phases(joints, phases, events, fps):
    n = len(joints["l_ankle"][0])
    scores, notes = {}, {}

    apex = events["gather_apex"]
    l_k = _knee_angle(joints, "l", apex)
    r_k = _knee_angle(joints, "r", apex)
    load_angle = min(l_k, r_k)
    scores["GATHER"] = _band_score(load_angle, 95, 145, soft=20)
    notes["GATHER"] = (f"Loading knee at about {load_angle:.0f} degrees of bend at the top of the jump -- "
                        "a clear, compact load." if scores["GATHER"] >= 7 else
                        f"Loading knee at about {load_angle:.0f} degrees -- either quite stiff or quite "
                        "collapsed at the top of the gather, worth a look.")

    bfc = events["back_foot_contact"]
    lo, hi = max(0, bfc - 1), min(n - 1, bfc + 1)
    land_sharpness = abs(joints["l_ankle"][1][hi] + joints["r_ankle"][1][hi]
                          - joints["l_ankle"][1][lo] - joints["r_ankle"][1][lo])
    torso = _torso_len(joints, bfc)
    scores["BACK FOOT CONTACT"] = max(1, min(10, round(4 + 6 * (land_sharpness / (torso * 0.6)))))
    notes["BACK FOOT CONTACT"] = "Firm, positive landing into the back foot." if scores["BACK FOOT CONTACT"] >= 7 \
        else "Landing detected but the signal is soft here -- treat as low-confidence."

    ffc = events["front_foot_contact"]
    l_k2 = _knee_angle(joints, "l", ffc)
    r_k2 = _knee_angle(joints, "r", ffc)
    brace_angle = max(l_k2, r_k2)
    scores["FRONT FOOT CONTACT"] = _band_score(brace_angle, 155, 178, soft=18)
    notes["FRONT FOOT CONTACT"] = (f"Front leg braced at roughly {brace_angle:.0f} degrees -- a firm block "
                                    "to rotate over." if scores["FRONT FOOT CONTACT"] >= 7 else
                                    f"Front leg at roughly {brace_angle:.0f} degrees -- looks like it's "
                                    "staying quite bent through landing rather than bracing up, which "
                                    "usually costs both pace and puts more load through the knee.")

    release = events["release"]
    bow_arm = events["bowling_arm"]
    sh_y = joints[f"{bow_arm}_shoulder"][1][release]
    wr_y = joints[f"{bow_arm}_wrist"][1][release]
    torso_r = _torso_len(joints, release)
    lift = (sh_y - wr_y) / torso_r
    elbow = _elbow_angle(joints, bow_arm, release)
    height_score = max(1, min(10, round(5 + lift * 6)))
    elbow_score = _band_score(elbow, 155, 180, soft=20)
    scores["RELEASE / SNAP"] = round((height_score * 0.6 + elbow_score * 0.4))
    notes["RELEASE / SNAP"] = (f"High release, arm well extended (elbow ~{elbow:.0f} degrees) -- a strong "
                                "release point." if scores["RELEASE / SNAP"] >= 7 else
                                f"Release point looks a little lower/more bent (elbow ~{elbow:.0f} degrees) "
                                "than ideal -- may be worth filming closer to the crease to confirm.")

    ft_end = n - 1

    def trunk_angle(i):
        shx = (joints["l_shoulder"][0][i] + joints["r_shoulder"][0][i]) / 2
        shy = (joints["l_shoulder"][1][i] + joints["r_shoulder"][1][i]) / 2
        hpx = (joints["l_hip"][0][i] + joints["r_hip"][0][i]) / 2
        hpy = (joints["l_hip"][1][i] + joints["r_hip"][1][i]) / 2
        return float(np.degrees(np.arctan2(shx - hpx, hpy - shy)))

    lean_release = trunk_angle(release)
    lean_end = trunk_angle(ft_end)
    continuation = abs(lean_end - lean_release)
    scores["FOLLOW-THROUGH"] = max(1, min(10, round(3 + continuation / 6)))
    notes["FOLLOW-THROUGH"] = "Long continuation through the crease, trunk kept rotating well after release." \
        if scores["FOLLOW-THROUGH"] >= 7 else \
        "Follow-through looks like it stops fairly abruptly, or the clip ends too soon after release to tell."

    scores["RUN-UP"] = None
    notes["RUN-UP"] = "Not scored yet -- run-up rhythm scoring is on the roadmap."

    return scores, notes
