#!/usr/bin/env python3
"""Export touch dream .npz episode data to web-friendly format.

Generates:
  cam_right.mp4  - Right half of stereo head cam video
  data.json      - Force + latent data per frame (compact)

Usage:
  python export_touch_dream_web.py \\
    --dir /path/to/step_*.npz_directory \\
    --output_dir touch_dream_data/insert_t_ep1 \\
    --subsample 3 --fps 30
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb", "palm"]
FINGER_RANGES = {
    "pinky": (0, 185),
    "ring": (185, 370),
    "middle": (370, 555),
    "index": (555, 740),
    "thumb": (740, 950),
    "palm": (950, 1062),
}


def round_list(arr, precision, clamp=None):
    if clamp is not None:
        lo, hi = clamp
        return [round(max(lo, min(hi, float(v))), precision) for v in arr]
    return [round(float(v), precision) for v in arr]


def detect_mode(data):
    for finger in FINGER_NAMES:
        key = f"dream_left_{finger}_tactile"
        if key in data:
            dim = data[key].shape[-1]
            expected = FINGER_RANGES[finger][1] - FINGER_RANGES[finger][0]
            return "raw" if dim == expected else "latent"
    return "unknown"


def export_cam_video(frames, output_path, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name
        for frame in frames:
            tmp.write(frame.astype(np.uint8).tobytes())

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", output_path,
    ]
    subprocess.run(cmd, capture_output=True)
    os.unlink(tmp_path)
    print(f"Saved cam video: {output_path} ({len(frames)} frames, {fps:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Export touch dream data for web visualization")
    parser.add_argument("--dir", required=True, help="Directory with step_*.npz files")
    parser.add_argument("--output_dir", required=True, help="Output directory for web data")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Frame stride (default 3 = 10Hz from 30Hz)")
    parser.add_argument("--precision", type=int, default=3,
                        help="Decimal precision for JSON floats")
    parser.add_argument("--fps", type=float, default=30.0, help="Original capture FPS")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "step_*.npz")))
    if not files:
        raise FileNotFoundError(f"No step_*.npz files in {args.dir}")

    windows = [np.load(f, allow_pickle=True) for f in files]
    steps_per_window = windows[0]["gt_left_force"].shape[0]
    total_raw_frames = len(windows) * steps_per_window
    mode = detect_mode(windows[0])
    print(f"Loaded {len(windows)} windows x {steps_per_window} steps = {total_raw_frames} frames")
    print(f"Mode: {mode}, subsample: {args.subsample}")

    os.makedirs(args.output_dir, exist_ok=True)

    frame_indices = list(range(0, total_raw_frames, args.subsample))
    out_fps = args.fps / args.subsample

    force = {"left_gt": [], "left_pred": [], "right_gt": [], "right_pred": []}
    latent = {}
    gt_raw = {"left": [], "right": []}
    raw_pred = {"left": [], "right": []}
    cam_frames = []

    has_force = False
    has_latent = False
    has_gt_raw = False
    has_raw_pred = False
    latent_dim = 0
    raw_prec = min(args.precision, 2)

    for fi in frame_indices:
        wi = fi // steps_per_window
        si = fi % steps_per_window
        data = windows[wi]

        for side in ("left", "right"):
            pred_key = f"dream_{side}_eef_finger_force"
            gt_key = f"gt_{side}_force"
            if pred_key in data and gt_key in data:
                has_force = True
                p = data[pred_key][si] if data[pred_key].ndim == 2 else data[pred_key]
                g = data[gt_key][si] if data[gt_key].ndim == 2 else data[gt_key]
                force[f"{side}_pred"].append(round_list(p, args.precision))
                force[f"{side}_gt"].append(round_list(g, args.precision))

        if mode == "latent":
            for side in ("left", "right"):
                for finger in FINGER_NAMES:
                    pk = f"dream_{side}_{finger}_tactile"
                    gk = f"gt_latent_{side}_{finger}_tactile"
                    if pk in data and gk in data:
                        has_latent = True
                        kp = f"{side}_{finger}_pred"
                        kg = f"{side}_{finger}_gt"
                        if kp not in latent:
                            latent[kp] = []
                            latent[kg] = []
                        p = data[pk][si] if data[pk].ndim == 2 else data[pk]
                        g = data[gk][si] if data[gk].ndim == 2 else data[gk]
                        latent[kp].append(round_list(p, args.precision))
                        latent[kg].append(round_list(g, args.precision))
                        if latent_dim == 0:
                            latent_dim = len(p)

        for side in ("left", "right"):
            gt_key = f"gt_{side}_tactile"
            if gt_key in data:
                has_gt_raw = True
                g = data[gt_key][si] if data[gt_key].ndim == 2 else data[gt_key]
                gt_raw[side].append(round_list(g, raw_prec, clamp=(0, 1)))

        if mode == "raw":
            for side in ("left", "right"):
                full_pred = np.zeros(1062, dtype=np.float32)
                for finger, (start, end) in FINGER_RANGES.items():
                    fk = f"dream_{side}_{finger}_tactile"
                    if fk in data:
                        has_raw_pred = True
                        fv = data[fk]
                        fv = fv[si] if fv.ndim == 2 else fv
                        full_pred[start:end] = fv
                raw_pred[side].append(round_list(full_pred, raw_prec, clamp=(0, 1)))

        cam = data.get("head_cam")
        if cam is not None:
            frame = cam[si] if cam.ndim == 4 else cam
            W = frame.shape[1]
            cam_frames.append(frame[:, W // 2:, :])

    all_force_vals = []
    for k in force:
        for arr in force[k]:
            all_force_vals.extend(arr)
    force_min = min(all_force_vals) if all_force_vals else 0
    force_max = max(all_force_vals) if all_force_vals else 1

    all_latent_vals = []
    for k in latent:
        for arr in latent[k]:
            all_latent_vals.extend(arr)
    latent_min = min(all_latent_vals) if all_latent_vals else 0
    latent_max = max(all_latent_vals) if all_latent_vals else 1

    raw_min = 0.0
    raw_max = 1.0

    output = {
        "meta": {
            "totalFrames": len(frame_indices),
            "fps": round(out_fps, 2),
            "mode": mode,
            "hasForce": has_force,
            "hasLatent": has_latent,
            "hasGtRaw": has_gt_raw,
            "hasRawPred": has_raw_pred,
            "latentDim": latent_dim,
            "fingerNames": FINGER_NAMES,
            "forceLabels": ["Th1", "Th2", "Index", "Middle", "Ring", "Pinky"],
            "forceMin": round(float(force_min) - 0.1, args.precision),
            "forceMax": round(float(force_max) + 0.1, args.precision),
            "latentMin": round(float(latent_min), args.precision),
            "latentMax": round(float(latent_max), args.precision),
            "rawMin": round(float(raw_min), raw_prec),
            "rawMax": round(float(raw_max), raw_prec),
        },
    }
    if has_force:
        output["force"] = force
    if has_latent:
        output["latent"] = latent
    if has_gt_raw:
        output["gt_raw"] = gt_raw
    if has_raw_pred:
        output["raw_pred"] = raw_pred

    json_path = os.path.join(args.output_dir, "data.json")
    with open(json_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    size_kb = os.path.getsize(json_path) / 1024
    print(f"Saved {json_path} ({size_kb:.0f} KB, {len(frame_indices)} frames)")

    if cam_frames:
        export_cam_video(cam_frames, os.path.join(args.output_dir, "cam_right.mp4"), out_fps)
    else:
        print("No head_cam data found in npz files, skipping cam video.")

    print("Done!")


if __name__ == "__main__":
    main()
