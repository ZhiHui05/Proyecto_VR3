"""
stick_tracker.py
----------------
Tracks a 5-faced ArUco cube mounted on a stick and reports the 6-DOF pose
(position + orientation) of the stick in real-time.

How it works
------------
1. Load the 5 marker IDs chosen by generate_markers.py (max Hamming distance).
2. Each frame: detect all ArUco markers and keep only the ones belonging to
   the stick cube.
3. Estimate the pose of every visible face, then compute the cube-centre pose
   using the known face→cube transform.
4. Average the per-face estimates to improve accuracy when multiple faces are
   visible simultaneously.
5. Express the stick orientation as Euler angles (roll / pitch / yaw).
6. If no cube marker is visible → "Tracking lost" overlay and keep waiting.

Coordinate conventions
----------------------
  Camera frame : X right, Y down, Z into the scene (standard OpenCV).
  Cube frame   : X right, Y up,   Z front.
  Stick axis   : the stick extends along +Y in cube frame (cube is at the top,
                  stick handle points downward).  Change STICK_AXIS_CUBE below
                  if your physical build is different.

Face-to-cube transforms
-----------------------
The cube's 5 marker faces and their orientation relative to the cube centre are
defined in the FACE_CONFIG dictionary.  Each entry holds:
  • rvec  – Rodrigues vector rotating the face frame INTO the cube frame.
  • tvec  – translation of the face centre from the cube centre (metres).

Adjust MARKER_SIZE and CUBE_SIDE to match your printed/built dimensions.

Camera calibration
------------------
Place a file named 'calibration.npz' (keys: 'camera_matrix', 'dist_coeffs')
in the same directory as this script.  If the file is not found the tracker
uses a rough pinhole estimate based on the frame resolution — good enough for
a first test but less accurate.

Controls
--------
  ESC   – quit
  Space – freeze / resume display (does NOT stop tracking)
"""

from __future__ import annotations
import cv2
import numpy as np
import sys
import os
import time
import math

# ── User-adjustable parameters ────────────────────────────────────────────────

MARKER_SIZE  = 0.05          # printed marker side length in metres (5 cm)
CUBE_SIDE    = 0.06          # physical cube side length in metres   (6 cm)
CAMERA_INDEX = 0             # webcam index (change if using a different camera)
TARGET_FPS   = 30            # desired capture frame-rate
ARUCO_DICT   = cv2.aruco.DICT_6X6_50

# Stick axis expressed in the CUBE reference frame.
# (0, 1, 0) means the stick extends along +Y (handle is below the cube when Y
#  is pointing downward in the physical world).  Flip to (0, -1, 0) if needed.
STICK_AXIS_CUBE = np.array([0.0, 1.0, 0.0])

# ── Internal constants ────────────────────────────────────────────────────────

HALF = CUBE_SIDE / 2.0      # shorthand

# Face configuration: face_id → (face_rvec, face_tvec_in_cube_frame)
#
# ArUco pose gives us the face frame where +Z is the *outward* face normal.
# rvec_face_to_cube brings that face frame to the cube frame.
# tvec is where the face centre sits inside the cube frame.
#
#  Face layout:
#   0 – FRONT  : normal = +Z_cube, offset = (0, 0, +HALF)
#   1 – BACK   : normal = −Z_cube, offset = (0, 0, −HALF)
#   2 – LEFT   : normal = −X_cube, offset = (−HALF, 0, 0)
#   3 – RIGHT  : normal = +X_cube, offset = (+HALF, 0, 0)
#   4 – TOP    : normal = +Y_cube, offset = (0, +HALF, 0)
#
# The BOTTOM face (where the stick is attached) has no marker.

def _make_face_config(half: float) -> dict:
    """Return a dict mapping face index → (rvec_face_to_cube, tvec_face_in_cube)."""
    configs = {}

    # FRONT face — marker +Z points in the same direction as cube +Z
    # No rotation needed between face frame and cube frame.
    configs[0] = (
        np.array([0.0, 0.0, 0.0]),          # rvec: identity
        np.array([0.0, 0.0, half])          # face centre is +half in Z
    )

    # BACK face — marker +Z points in –Z_cube direction
    # Rotate 180° around Y to flip Z.
    configs[1] = (
        np.array([0.0, math.pi, 0.0]),
        np.array([0.0, 0.0, -half])
    )

    # LEFT face — marker +Z points in –X_cube direction
    # Rotate –90° around Y so that face's +Z aligns with –X_cube.
    configs[2] = (
        np.array([0.0, -math.pi / 2, 0.0]),
        np.array([-half, 0.0, 0.0])
    )

    # RIGHT face — marker +Z points in +X_cube direction
    # Rotate +90° around Y.
    configs[3] = (
        np.array([0.0, math.pi / 2, 0.0]),
        np.array([half, 0.0, 0.0])
    )

    # TOP face — marker +Z points in +Y_cube direction
    # Rotate –90° around X so face's +Z aligns with +Y_cube.
    configs[4] = (
        np.array([-math.pi / 2, 0.0, 0.0]),
        np.array([0.0, half, 0.0])
    )

    # Convert rvecs to rotation matrices for convenience
    result = {}
    for face_idx, (rvec, tvec) in configs.items():
        R, _ = cv2.Rodrigues(rvec)
        result[face_idx] = {"R_face_to_cube": R, "t_face_in_cube": tvec}
    return result


FACE_CONFIG = _make_face_config(HALF)

# ── Helper functions ──────────────────────────────────────────────────────────

def load_camera_calibration(frame_w: int, frame_h: int):
    """
    Try to load calibration from 'calibration.npz'.  Fall back to a
    default pinhole estimate if the file is not found.
    """
    calib_path = os.path.join(os.path.dirname(__file__), "calibration.npz")
    if os.path.isfile(calib_path):
        data = np.load(calib_path)
        camera_matrix = data["camera_matrix"]
        dist_coeffs   = data["dist_coeffs"]
        print(f"[CALIB] Loaded calibration from {calib_path}")
    else:
        # Approximate focal length: f ≈ 0.8 × image width (heuristic for webcam)
        f  = 0.8 * max(frame_w, frame_h)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        camera_matrix = np.array([[f, 0, cx],
                                   [0, f, cy],
                                   [0, 0,  1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        print("[CALIB] WARNING: no 'calibration.npz' found — using default "
              "pinhole estimate.  Results may be inaccurate.")
    return camera_matrix, dist_coeffs


def load_target_ids() -> list[int]:
    """
    Load marker IDs from markers/selected_ids.txt (written by generate_markers.py).
    Fall back to [0, 1, 2, 3, 4] if the file is not found.
    """
    id_file = os.path.join(os.path.dirname(__file__), "markers", "selected_ids.txt")
    if os.path.isfile(id_file):
        with open(id_file) as f:
            ids = [int(x.strip()) for x in f.read().split(",") if x.strip()]
        print(f"[IDs]   Loaded target marker IDs from file: {ids}")
    else:
        ids = list(range(5))
        print(f"[IDs]   'selected_ids.txt' not found — using default IDs: {ids}")
    return ids


def rvec_tvec_to_matrix(rvec, tvec) -> np.ndarray:
    """Return a 4×4 homogeneous transform from rvec and tvec."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.flatten()
    return T


def matrix_to_rvec_tvec(T: np.ndarray):
    """Split a 4×4 homogeneous matrix back into rvec and tvec."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec    = T[:3, 3]
    return rvec.flatten(), tvec.flatten()


def face_pose_to_cube_pose(face_idx: int, rvec_face, tvec_face):
    """
    Given the detected pose of a face (rvec_face, tvec_face in camera frame),
    compute the pose of the CUBE CENTRE in the camera frame.

    The relationship is:
        T_cube_in_cam = T_face_in_cam  ×  T_face_in_cube^(-1)

    Where T_face_in_cube is the known face→cube transform in cube frame
    (i.e. T_cube_in_face inverted).

    More precisely:
        T_face_in_cam = T_cam_to_face (camera to face-origin)
    But ArUco gives us T_face_in_cam directly as {R_face, t_face}.

        T_cube_in_cam = T_face_in_cam * inv(T_face_relative_to_cube)
    """
    cfg = FACE_CONFIG[face_idx]
    R_fc  = cfg["R_face_to_cube"]        # rotation: face frame → cube frame
    t_fc  = cfg["t_face_in_cube"]        # translation in cube frame

    # Build T_face_in_cube  (pose of face expressed in cube coords)
    T_face_in_cube = np.eye(4)
    T_face_in_cube[:3, :3] = R_fc
    T_face_in_cube[:3, 3]  = t_fc

    # Build T_face_in_cam  (pose of face in camera coords — from ArUco)
    T_face_in_cam = rvec_tvec_to_matrix(rvec_face, tvec_face)

    # Cube in camera = (face in cam) × inverse(face in cube)
    T_cube_in_cam = T_face_in_cam @ np.linalg.inv(T_face_in_cube)
    return T_cube_in_cam


def average_rotation_matrices(Rs: list[np.ndarray]) -> np.ndarray:
    """
    Average multiple rotation matrices using the SVD-based geodesic mean
    (project the element-wise mean back onto SO(3)).
    """
    mean = np.mean(np.array(Rs), axis=0)
    U, _, Vt = np.linalg.svd(mean)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose a rotation matrix into Euler angles (roll, pitch, yaw) in
    degrees using the ZYX convention (yaw → pitch → roll).
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.degrees(math.atan2( R[2, 1],  R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = math.degrees(math.atan2( R[1, 0],  R[0, 0]))
    else:
        roll  = math.degrees(math.atan2(-R[1, 2],  R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0],  sy))
        yaw   = 0.0
    return roll, pitch, yaw


def draw_hud(frame: np.ndarray,
             pos: np.ndarray | None,
             orientation: tuple | None,
             stick_dir: np.ndarray | None,
             tracking: bool,
             fps: float,
             n_faces_visible: int) -> None:
    """Overlay tracking info on the frame (in-place)."""
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.55
    big   = 0.75
    green = (50, 220, 50)
    red   = (0, 60, 220)
    white = (230, 230, 230)
    grey  = (120, 120, 120)

    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "ArUco Stick Tracker", (10, 25),
                font, big, white, 2)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Faces visible: {n_faces_visible}",
                (10, 52), font, small, grey, 1)

    if tracking and pos is not None and orientation is not None:
        roll, pitch, yaw = orientation
        px, py, pz = pos
        cv2.putText(frame,
                    f"Position  X:{px:+7.3f}  Y:{py:+7.3f}  Z:{pz:+7.3f} m",
                    (10, 80), font, small, green, 1)
        cv2.putText(frame,
                    f"Rotation  Roll:{roll:+7.1f}  Pitch:{pitch:+7.1f}  Yaw:{yaw:+7.1f} deg",
                    (10, 105), font, small, green, 1)
        if stick_dir is not None:
            dx, dy, dz = stick_dir
            cv2.putText(frame,
                        f"Stick dir  [{dx:+.3f}, {dy:+.3f}, {dz:+.3f}]",
                        (10, 128), font, small, (200, 200, 50), 1)
    else:
        # Blinking red "Tracking lost" message
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, "⚠ TRACKING LOST — awaiting stick…",
                        (10, 90), font, big, red, 3)
            cv2.putText(frame, "⚠ TRACKING LOST — awaiting stick…",
                        (10, 90), font, big, (30, 30, 200), 1)


def draw_cube_axes(frame, camera_matrix, dist_coeffs,
                   T_cube_in_cam: np.ndarray) -> None:
    """Draw a 3-axis gizmo at the estimated cube centre."""
    rvec, tvec = matrix_to_rvec_tvec(T_cube_in_cam)
    length = CUBE_SIDE * 1.5
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                      rvec, tvec, length, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── ArUco setup ───────────────────────────────────────────────────────────
    dictionary   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    det_params   = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(dictionary, det_params)
    target_ids   = load_target_ids()
    target_id_set = set(target_ids)

    # Map: marker_id → face_index  (the face this marker lives on)
    # The order in target_ids matches face 0–4.
    id_to_face = {mid: face for face, mid in enumerate(target_ids)}

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    ret, frame0 = cap.read()
    if not ret:
        print("[ERROR] Cannot read from camera.")
        sys.exit(1)

    h0, w0 = frame0.shape[:2]
    camera_matrix, dist_coeffs = load_camera_calibration(w0, h0)

    print("[INFO]  Press ESC to quit.  Press SPACE to freeze/resume display.")
    print(f"[INFO]  Tracking marker IDs: {target_ids}")
    print(f"[INFO]  Marker size: {MARKER_SIZE*100:.0f} cm | "
          f"Cube side: {CUBE_SIDE*100:.0f} cm\n")

    # ── State ─────────────────────────────────────────────────────────────────
    paused        = False
    tick_freq     = cv2.getTickFrequency()
    fps_display   = 0.0
    was_tracking  = False       # used to print status changes to console
    last_lost_msg = 0.0         # throttle console "tracking lost" messages

    while True:
        t_start = cv2.getTickCount()

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost camera feed.")
            break

        if paused:
            cv2.imshow("ArUco Stick Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                paused = False
            continue

        # ── Detection ─────────────────────────────────────────────────────────
        corners, ids, _ = detector.detectMarkers(frame)

        # Filter: keep only markers that belong to the stick
        cube_rvecs = []
        cube_tvecs = []
        n_visible  = 0
        T_cube_estimates = []

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid not in target_id_set:
                    continue
                face_idx = id_to_face[mid]
                n_visible += 1

                # Estimate face pose
                rvecs_f, tvecs_f, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], MARKER_SIZE, camera_matrix, dist_coeffs)

                rvec_f = rvecs_f[0][0]
                tvec_f = tvecs_f[0][0]

                # Draw marker outline and face ID
                cv2.aruco.drawDetectedMarkers(frame, [corners[i]])
                cv2.putText(frame, f"F{face_idx}",
                            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                # Convert face pose → cube-centre pose
                T_cube = face_pose_to_cube_pose(face_idx, rvec_f, tvec_f)
                T_cube_estimates.append(T_cube)

        # ── Fuse estimates from multiple visible faces ─────────────────────
        tracking = len(T_cube_estimates) > 0
        pos       = None
        euler     = None
        stick_dir = None
        T_final   = None

        if tracking:
            if len(T_cube_estimates) == 1:
                T_final = T_cube_estimates[0]
            else:
                Rs    = [T[:3, :3] for T in T_cube_estimates]
                ts    = [T[:3, 3]  for T in T_cube_estimates]
                R_avg = average_rotation_matrices(Rs)
                t_avg = np.mean(np.array(ts), axis=0)
                T_final = np.eye(4)
                T_final[:3, :3] = R_avg
                T_final[:3, 3]  = t_avg

            pos       = T_final[:3, 3]
            R_cube    = T_final[:3, :3]
            euler     = rotation_matrix_to_euler(R_cube)

            # Stick direction in camera frame
            stick_dir = R_cube @ STICK_AXIS_CUBE

            # Draw cube-centre axes gizmo
            draw_cube_axes(frame, camera_matrix, dist_coeffs, T_final)

        # ── Console status changes ────────────────────────────────────────────
        if tracking and not was_tracking:
            print(f"[STATUS] Stick acquired!  {n_visible} face(s) visible.")
            was_tracking = True
        elif not tracking and was_tracking:
            print("[STATUS] Tracking LOST — waiting for stick to reappear…")
            was_tracking = False
        elif not tracking:
            now = time.time()
            if now - last_lost_msg > 3.0:
                print("[STATUS] Still waiting for stick…")
                last_lost_msg = now

        # ── Print pose to console (every frame, only when tracking) ───────────
        if tracking and pos is not None and euler is not None:
            roll, pitch, yaw = euler
            dx, dy, dz = stick_dir
            print(f"\r[POSE]  pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})m  "
                  f"roll={roll:+6.1f}° pitch={pitch:+6.1f}° yaw={yaw:+6.1f}°  "
                  f"stick=[{dx:+.3f},{dy:+.3f},{dz:+.3f}]  faces={n_visible}",
                  end="", flush=True)

        # ── HUD overlay ───────────────────────────────────────────────────────
        fps_display = tick_freq / (cv2.getTickCount() - t_start)
        draw_hud(frame, pos, euler, stick_dir, tracking, fps_display, n_visible)

        cv2.imshow("ArUco Stick Tracker", frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:            # ESC → quit
            print("\n[INFO]  Quitting…")
            break
        if key == 32:            # SPACE → freeze
            paused = True
            print("\n[INFO]  Paused.  Press SPACE to resume.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
