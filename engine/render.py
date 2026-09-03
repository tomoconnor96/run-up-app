import cv2
import subprocess
import os

PHASE_COLORS = {
    "RUN-UP": (200, 200, 200),
    "GATHER": (0, 215, 255),
    "BACK FOOT CONTACT": (0, 140, 255),
    "STRIDE": (180, 180, 180),
    "FRONT FOOT CONTACT": (0, 80, 255),
    "RELEASE / SNAP": (0, 0, 255),
    "FOLLOW-THROUGH": (0, 200, 0),
}


def render_annotated(frames, fps, phases, events, out_path_raw, out_path_final, video_path):
    H, W = frames[0].shape[:2]
    n = len(frames)

    def phase_for(i):
        for name, a, b in phases:
            if a <= i <= b:
                return name
        return ""

    event_frame_labels = {}
    if events.get("back_foot_contact") is not None:
        event_frame_labels[events["back_foot_contact"]] = "BFC!"
    if events.get("front_foot_contact") is not None:
        event_frame_labels[events["front_foot_contact"]] = "FFC!"
    if events.get("release") is not None:
        event_frame_labels[events["release"]] = "RELEASE!"

    out_frames = []
    for i, frame in enumerate(frames):
        f = frame.copy()
        name = phase_for(i)
        color = PHASE_COLORS.get(name, (255, 255, 255))

        cv2.rectangle(f, (0, 0), (W, 46), (20, 20, 20), -1)
        cv2.putText(f, name, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        strip_y0 = H - 26
        seg_w = W / len(phases)
        for idx, (pname, a, b) in enumerate(phases):
            x0 = int(idx * seg_w)
            x1 = int((idx + 1) * seg_w)
            active = a <= i <= b
            done = i > b
            col = PHASE_COLORS.get(pname, (255, 255, 255)) if (active or done) else (70, 70, 70)
            cv2.rectangle(f, (x0, strip_y0), (x1 - 2, H - 4), col, -1 if active else 1)

        if i in event_frame_labels:
            cv2.rectangle(f, (4, 50), (W - 4, H - 32), (0, 0, 255), 6)
            cv2.putText(f, event_frame_labels[i], (max(10, W // 2 - 90), H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        t = i / fps
        cv2.putText(f, f"t={t:.2f}s f{i}", (W - 165, H - 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out_frames.append(f)

    apex = events.get("gather_apex", n // 3)
    slow_lo, slow_hi = max(0, apex - 5), min(n - 1, apex + 25)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path_raw, fourcc, fps, (W, H))
    for i, f in enumerate(out_frames):
        reps = 5 if slow_lo <= i <= slow_hi else 3
        for _ in range(reps):
            writer.write(f)
    writer.release()

    cmd = [
        "ffmpeg", "-y", "-i", out_path_raw, "-i", video_path,
        "-filter_complex", "[1:a]atempo=0.5[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-shortest",
        out_path_final,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path_final):
        # fall back to no-audio version if source has no/odd audio stream
        cmd2 = ["ffmpeg", "-y", "-i", out_path_raw, "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "20", out_path_final]
        subprocess.run(cmd2, capture_output=True, text=True)
    return out_path_final
