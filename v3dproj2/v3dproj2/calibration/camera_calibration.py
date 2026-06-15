#!/usr/bin/env python3
"""
Camera calibration utility using OpenCV chessboard pattern.

Detects chessboard corners from camera frames or image files,
computes the camera intrinsic matrix and distortion coefficients,
and saves calibration results to disk.

Usage:
    python -m calibration.camera_calibration --device 0
    python -m calibration.camera_calibration --images data/calibration_images/
    python -m calibration.camera_calibration --device 0 --output data/calibration/
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class CameraCalibrator:
    """
    Performs camera calibration using a chessboard pattern.

    Captures or loads chessboard images, detects corners,
    and computes intrinsic and distortion parameters.
    """

    def __init__(
        self,
        chessboard_rows: int = 9,
        chessboard_cols: int = 6,
        square_size_mm: float = 25.0,
    ) -> None:
        self._chessboard_dims = (chessboard_cols, chessboard_rows)
        self._square_size = square_size_mm
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        self._objpoints: List[np.ndarray] = []
        self._imgpoints: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None

        self._objp = np.zeros(
            (chessboard_cols * chessboard_rows, 3), dtype=np.float32
        )
        self._objp[:, :2] = np.mgrid[
            0:chessboard_cols, 0:chessboard_rows
        ].T.reshape(-1, 2) * square_size_mm

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._rvecs: List[np.ndarray] = []
        self._tvecs: List[np.ndarray] = []
        self._reprojection_error: float = 0.0

    def capture_from_camera(
        self,
        device_index: int = 0,
        num_images: int = 20,
        delay: float = 0.5,
    ) -> bool:
        """
        Capture chessboard images from a live camera feed.

        Press SPACE to capture, 'q' to quit early.
        """
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera device {device_index}")
            return False

        print(f"\n{'='*60}")
        print("  CAMERA CALIBRATION - Live Capture Mode")
        print(f"  Device: {device_index}")
        print(f"  Chessboard: {self._chessboard_dims[0]}x{self._chessboard_dims[1]}")
        print(f"  Target images: {num_images}")
        print(f"{'='*60}")
        print("\n  SPACE = Capture frame")
        print("  q/ESC = Finish and calibrate")
        print("")

        captured = 0

        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._image_size is None:
                self._image_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray, self._chessboard_dims, None
            )

            display = frame.copy()
            if found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self._criteria
                )
                cv2.drawChessboardCorners(
                    display, self._chessboard_dims, refined, found
                )
                status = "CHESSBOARD DETECTED"
                color = (0, 255, 0)
            else:
                status = "No chessboard detected"
                color = (0, 0, 255)

            cv2.putText(
                display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            cv2.putText(
                display, f"Captured: {captured}/{num_images}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            cv2.putText(
                display, "SPACE=capture  q=quit",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            cv2.imshow("Camera Calibration", display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord(" ") and found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self._criteria
                )
                self._objpoints.append(self._objp.copy())
                self._imgpoints.append(refined)
                captured += 1
                print(f"  Captured [{captured}/{num_images}]")
                time.sleep(delay)
            elif key == ord("q") or key == 27:
                print(f"  Capture ended early with {captured} images")
                break

            time.sleep(0.03)

        cap.release()
        cv2.destroyAllWindows()

        if captured < 5:
            print(f"WARNING: Only {captured} images captured. At least 5 recommended.")

        return self._perform_calibration()

    def process_images(self, image_dir: str) -> bool:
        """
        Load chessboard images from a directory and calibrate.

        Args:
            image_dir: Path to directory containing chessboard images.
        """
        img_path = Path(image_dir)
        if not img_path.is_dir():
            print(f"ERROR: Directory not found: {image_dir}")
            return False

        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        image_files: List[Path] = []
        for ext in image_extensions:
            image_files.extend(img_path.glob(ext))

        if not image_files:
            print(f"ERROR: No images found in {image_dir}")
            return False

        print(f"\n{'='*60}")
        print("  CAMERA CALIBRATION - Image Processing Mode")
        print(f"  Found {len(image_files)} images")
        print(f"  Chessboard: {self._chessboard_dims[0]}x{self._chessboard_dims[1]}")
        print(f"{'='*60}\n")

        for img_file in image_files:
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"  WARNING: Cannot read {img_file.name}")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._image_size is None:
                self._image_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray, self._chessboard_dims, None
            )

            if found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self._criteria
                )
                self._objpoints.append(self._objp.copy())
                self._imgpoints.append(refined)
                print(f"  [{len(self._imgpoints):3d}] OK - {img_file.name}")
            else:
                print(f"  [    ] FAIL - {img_file.name} (no chessboard found)")

        if len(self._imgpoints) < 5:
            print(f"\nWARNING: Only found {len(self._imgpoints)} valid images. Need at least 5.")
            return False

        return self._perform_calibration()

    def _perform_calibration(self) -> bool:
        if not self._imgpoints or self._image_size is None:
            print("ERROR: No calibration data collected")
            return False

        print(f"\n{'='*60}")
        print(f"  Calibrating with {len(self._imgpoints)} images...")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._objpoints,
            self._imgpoints,
            self._image_size,
            None,
            None,
        )

        self._camera_matrix = mtx
        self._dist_coeffs = dist
        self._rvecs = rvecs
        self._tvecs = tvecs

        total_error = 0.0
        for i in range(len(self._objpoints)):
            projected, _ = cv2.projectPoints(
                self._objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(
                self._imgpoints[i], projected, cv2.NORM_L2
            ) / len(projected)
            total_error += error

        self._reprojection_error = total_error / max(len(self._objpoints), 1)

        print(f"  RMS Re-projection Error: {self._reprojection_error:.4f} px")
        print(f"  Camera Matrix:\n{self._camera_matrix}")
        print(f"  Distortion Coeffs:\n{self._dist_coeffs.ravel()}")
        print(f"{'='*60}\n")

        return True

    def save(self, output_dir: str = "data/calibration") -> Optional[Path]:
        """
        Save calibration results to disk.

        Args:
            output_dir: Directory to save calibration files.

        Returns:
            Path to saved file, or None on failure.
        """
        if self._camera_matrix is None or self._dist_coeffs is None:
            print("ERROR: No calibration data to save")
            return None

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        calibration_data: Dict[str, Any] = {
            "camera_matrix": self._camera_matrix.tolist(),
            "dist_coeffs": self._dist_coeffs.tolist(),
            "image_size": list(self._image_size) if self._image_size else None,
            "reprojection_error": self._reprojection_error,
            "chessboard_dims": list(self._chessboard_dims),
            "square_size_mm": self._square_size,
            "num_images": len(self._imgpoints),
        }

        npz_path = out_path / "calibration.npz"
        np.savez(
            str(npz_path),
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            image_size=np.array(self._image_size),
        )
        print(f"  Saved: {npz_path}")

        pkl_path = out_path / "calibration.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(calibration_data, f)
        print(f"  Saved: {pkl_path}")

        txt_path = out_path / "calibration_report.txt"
        with open(txt_path, "w") as f:
            f.write("=== Camera Calibration Report ===\n\n")
            f.write(f"Chessboard: {self._chessboard_dims[0]}x{self._chessboard_dims[1]}\n")
            f.write(f"Square Size: {self._square_size} mm\n")
            f.write(f"Image Size: {self._image_size}\n")
            f.write(f"Number of images: {len(self._imgpoints)}\n")
            f.write(f"RMS Reprojection Error: {self._reprojection_error:.4f} px\n\n")
            f.write("Camera Matrix:\n")
            f.write(f"{self._camera_matrix}\n\n")
            f.write("Distortion Coefficients:\n")
            f.write(f"{self._dist_coeffs.ravel()}\n")
        print(f"  Saved: {txt_path}")

        return npz_path

    def load(self, calibration_path: str) -> bool:
        """
        Load calibration results from a saved file.

        Args:
            calibration_path: Path to .npz or .pkl calibration file.
        """
        path = Path(calibration_path)
        if not path.exists():
            print(f"ERROR: Calibration file not found: {calibration_path}")
            return False

        if path.suffix == ".npz":
            data = np.load(str(path))
            self._camera_matrix = data["camera_matrix"]
            self._dist_coeffs = data["dist_coeffs"]
            self._image_size = tuple(data["image_size"])
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._camera_matrix = np.array(data["camera_matrix"])
            self._dist_coeffs = np.array(data["dist_coeffs"])
            self._image_size = (
                tuple(data["image_size"]) if data.get("image_size") else None
            )
        else:
            print(f"ERROR: Unsupported file format: {path.suffix}")
            return False

        print(f"Loaded calibration from: {calibration_path}")
        print(f"  Camera Matrix:\n{self._camera_matrix}")
        return True

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply undistortion to a frame using loaded calibration.

        Args:
            frame: BGR image to undistort.

        Returns:
            Undistorted image.
        """
        if self._camera_matrix is None or self._dist_coeffs is None:
            return frame

        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix, self._dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            frame, self._camera_matrix, self._dist_coeffs, None, new_mtx
        )
        return undistorted

    @property
    def camera_matrix(self) -> Optional[np.ndarray]:
        return self._camera_matrix

    @property
    def dist_coeffs(self) -> Optional[np.ndarray]:
        return self._dist_coeffs

    @property
    def reprojection_error(self) -> float:
        return self._reprojection_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera calibration using chessboard pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m calibration.camera_calibration --device 0
  python -m calibration.camera_calibration --images data/calibration_images/
  python -m calibration.camera_calibration --device 0 --rows 9 --cols 6
        """,
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Camera device index for live capture",
    )
    parser.add_argument(
        "--images", type=str, default=None,
        help="Directory containing chessboard images",
    )
    parser.add_argument(
        "--rows", type=int, default=9,
        help="Number of inner corners per chessboard column (default: 9)",
    )
    parser.add_argument(
        "--cols", type=int, default=6,
        help="Number of inner corners per chessboard row (default: 6)",
    )
    parser.add_argument(
        "--square-size", type=float, default=25.0,
        help="Square size in millimeters (default: 25.0)",
    )
    parser.add_argument(
        "--output", type=str, default="data/calibration",
        help="Output directory for calibration files",
    )
    parser.add_argument(
        "--num-images", type=int, default=20,
        help="Number of images to capture from camera (default: 20)",
    )
    args = parser.parse_args()

    if args.device is None and args.images is None:
        print("ERROR: Must specify either --device or --images")
        parser.print_help()
        sys.exit(1)

    calibrator = CameraCalibrator(
        chessboard_rows=args.rows,
        chessboard_cols=args.cols,
        square_size_mm=args.square_size,
    )

    success = False
    if args.images:
        success = calibrator.process_images(args.images)
    elif args.device is not None:
        success = calibrator.capture_from_camera(
            device_index=args.device,
            num_images=args.num_images,
        )

    if success:
        calibrator.save(args.output)
        print("\nCalibration completed successfully!")
        print(f"Files saved to: {args.output}")
    else:
        print("\nCalibration failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
