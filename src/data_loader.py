import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_screen_calibration(data_dir: str, subj_id: str):
    """Loads physical screen dimensions from Calibration/screenSize.mat if available."""
    matches = glob.glob(os.path.join(data_dir, "**", subj_id, "**", "screenSize.mat"), recursive=True)
    if matches:
        mat = loadmat(matches[0])
        return {
            "w_px": float(mat["width_pixel"].squeeze()),
            "h_px": float(mat["height_pixel"].squeeze()),
            "w_mm": float(mat["width_mm"].squeeze()),
            "h_mm": float(mat["height_mm"].squeeze()),
        }
    return {"w_px": 1280.0, "h_px": 800.0, "w_mm": 286.0, "h_mm": 179.0}


def load_mpiigaze_dataset(data_dir: str = "./data") -> pd.DataFrame:
    """
    Loads real MPIIGaze data using the official README specification:
      - Annotation Subset/pXX.txt: img_path (e.g. day13/0203.jpg) + 16 manual coords (4 eye corners, 2 mouth, 2 pupils)
      - Data/Original/pXX/dayXX/annotation.txt:
          Dims 1~24 (0:24): detected eye landmarks
          Dims 25~26 (24:26): On-screen gaze target position (x, y) in pixels
          Dims 27~29 (26:29): 3D gaze target position
          Dims 30~35 (29:35): Estimated 3D head pose (rotation & translation)
    """
    anno_subset_files = sorted(glob.glob(os.path.join(data_dir, "**", "Annotation Subset", "p[0-9][0-9].txt"), recursive=True))
    orig_anno_files = glob.glob(os.path.join(data_dir, "**", "p[0-9][0-9]", "day*", "annotation.txt"), recursive=True)

    # Index Data/Original/pXX/dayYY/annotation.txt by (subj_id, day_id)
    day_anno_map = {}
    for oaf in orig_anno_files:
        parts = os.path.normpath(oaf).split(os.sep)
        day_id = parts[-2]   # e.g., day13
        subj_id = parts[-3]  # e.g., p00
        with open(oaf, "r") as f:
            lines = [line.strip().split() for line in f if len(line.strip().split()) >= 35]
        day_anno_map[(subj_id, day_id)] = lines

    records = []

    # Case 1: Join manual Annotation Subset landmarks with Data/Original/pXX/dayYY/annotation.txt targets (Dims 25-26 & 30-35)
    if anno_subset_files and day_anno_map:
        for af in anno_subset_files:
            subj_id = os.path.splitext(os.path.basename(af))[0]
            screen_info = load_screen_calibration(data_dir, subj_id)
            px_to_mm_x = screen_info["w_mm"] / screen_info["w_px"]
            px_to_mm_y = screen_info["h_mm"] / screen_info["h_px"]

            with open(af, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 17:
                        continue
                    img_key = parts[0].replace("\\", "/")  # e.g., "day13/0203.jpg"
                    day_id, img_name = img_key.split("/")[0], os.path.splitext(img_key.split("/")[1])[0]
                    line_idx = int(img_name) - 1  # 0203.jpg -> 0-based index 202

                    if (subj_id, day_id) not in day_anno_map:
                        continue
                    day_lines = day_anno_map[(subj_id, day_id)]
                    if line_idx < 0 or line_idx >= len(day_lines):
                        continue

                    orig_vals = day_lines[line_idx]
                    target_px = np.array(orig_vals[24:26], dtype=np.float64)  # Dims 25~26: screen (x, y)
                    head_pose = np.array(orig_vals[29:35], dtype=np.float64)  # Dims 30~35: 3D head pose

                    coords = np.array(parts[1:17], dtype=np.float64)
                    landmarks = np.zeros(24, dtype=np.float64)
                    landmarks[0:4] = coords[0:4]      # L_outer, L_inner
                    landmarks[4:6] = coords[12:14]    # L_pupil
                    landmarks[6:10] = coords[4:8]     # R_inner, R_outer
                    landmarks[10:12] = coords[14:16]  # R_pupil
                    landmarks[12:16] = coords[8:12]   # Mouth corners

                    target_mm = np.array([target_px[0] * px_to_mm_x, target_px[1] * px_to_mm_y], dtype=np.float64)
                    records.append({
                        "subject": subj_id,
                        "landmarks": landmarks,
                        "head_pose": head_pose,
                        "target_px": target_px,
                        "target_mm": target_mm,
                        "w_px": screen_info["w_px"],
                        "h_px": screen_info["h_px"],
                        "w_mm": screen_info["w_mm"],
                        "h_mm": screen_info["h_mm"],
                    })

    # Case 2: If only Data/Original/pXX/dayYY/annotation.txt is present, parse Dims 1~24, 25~26, 30~35 directly
    elif day_anno_map:
        for (subj_id, day_id), day_lines in sorted(day_anno_map.items()):
            screen_info = load_screen_calibration(data_dir, subj_id)
            px_to_mm_x = screen_info["w_mm"] / screen_info["w_px"]
            px_to_mm_y = screen_info["h_mm"] / screen_info["h_px"]
            # Subsample up to 700 frames per subject so all 15 subjects run fast
            for orig_vals in day_lines[::max(1, len(day_lines) // 50)]:
                raw24 = np.array(orig_vals[:24], dtype=np.float64)
                # Convert 12 eye landmarks (6 left eye, 6 right eye) into outer/inner/pupil layout
                l_pts = raw24[0:12].reshape(6, 2)
                r_pts = raw24[12:24].reshape(6, 2)
                landmarks = np.zeros(24, dtype=np.float64)
                landmarks[0:2] = l_pts[0]               # L_outer
                landmarks[2:4] = l_pts[3]               # L_inner
                landmarks[4:6] = l_pts.mean(axis=0)     # L_center/pupil
                landmarks[6:8] = r_pts[0]               # R_inner
                landmarks[8:10] = r_pts[3]              # R_outer
                landmarks[10:12] = r_pts.mean(axis=0)   # R_center/pupil

                target_px = np.array(orig_vals[24:26], dtype=np.float64)
                head_pose = np.array(orig_vals[29:35], dtype=np.float64)
                target_mm = np.array([target_px[0] * px_to_mm_x, target_px[1] * px_to_mm_y], dtype=np.float64)

                records.append({
                    "subject": subj_id,
                    "landmarks": landmarks,
                    "head_pose": head_pose,
                    "target_px": target_px,
                    "target_mm": target_mm,
                    "w_px": screen_info["w_px"],
                    "h_px": screen_info["h_px"],
                    "w_mm": screen_info["w_mm"],
                    "h_mm": screen_info["h_mm"],
                })

    if not records:
        raise RuntimeError(
            "Could not find Data/Original/pXX/dayXX/annotation.txt in ./data! "
            "Upload mpiigaze_complete_text.zip containing Data/Original/pXX/dayXX/annotation.txt."
        )

    df = pd.DataFrame(records)
    print(f"[+] Loaded {len(df)} real MPIIGaze frames across {df['subject'].nunique()} participants with true screen (x, y) targets!")
    return df
