from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PlayerHeatmapGenerator:
    def __init__(self, resize_width=1280):
        self.resize_width = resize_width

    def generate_heatmap(
        self,
        video_path,
        tracking_csv_path,
        heatmap_output_path,
        overlay_output_path,
        min_frames_seen=30,
        bins=80,
    ):
        video_path = Path(video_path)
        tracking_csv_path = Path(tracking_csv_path)
        heatmap_output_path = Path(heatmap_output_path)
        overlay_output_path = Path(overlay_output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if not tracking_csv_path.exists():
            raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")

        heatmap_output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_output_path.parent.mkdir(parents=True, exist_ok=True)

        tracking_df = pd.read_csv(tracking_csv_path)

        if tracking_df.empty:
            raise ValueError("Tracking CSV is empty.")

        frame = self._get_first_frame(video_path)
        frame = self._resize_frame(frame)
        height, width = frame.shape[:2]

        # Keep only stable player IDs
        valid_track_ids = (
            tracking_df.groupby("track_id")["frame"]
            .nunique()
            .reset_index(name="frames_seen")
        )

        valid_track_ids = valid_track_ids[
            valid_track_ids["frames_seen"] >= min_frames_seen
        ]["track_id"].tolist()

        filtered_df = tracking_df[
            tracking_df["track_id"].isin(valid_track_ids)
        ].copy()

        if filtered_df.empty:
            raise ValueError(
                "No valid tracks after filtering. Lower min_frames_seen."
            )

        x_points = filtered_df["center_x"].values
        y_points = filtered_df["center_y"].values

        print("\nGenerating player heatmap...")
        print(f"Total tracking points: {len(tracking_df)}")
        print(f"Filtered tracking points: {len(filtered_df)}")
        print(f"Valid player IDs: {len(valid_track_ids)}")
        print(f"Frame size: {width}x{height}")

        # -----------------------------
        # 1. Save standalone heatmap
        # -----------------------------
        plt.figure(figsize=(12, 7))
        plt.hist2d(
            x_points,
            y_points,
            bins=bins,
            range=[[0, width], [0, height]],
            cmap="hot",
        )

        plt.gca().invert_yaxis()
        plt.colorbar(label="Player Presence Density")
        plt.title("Player Movement Heatmap")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.tight_layout()
        plt.savefig(heatmap_output_path, dpi=200)
        plt.close()

        # -----------------------------
        # 2. Save heatmap overlay image
        # -----------------------------
        heatmap, xedges, yedges = np.histogram2d(
            x_points,
            y_points,
            bins=bins,
            range=[[0, width], [0, height]],
        )

        heatmap = heatmap.T

        heatmap_normalized = cv2.normalize(
            heatmap,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

        heatmap_uint8 = np.uint8(heatmap_normalized)

        heatmap_resized = cv2.resize(
            heatmap_uint8,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

        heatmap_colored = cv2.applyColorMap(
            heatmap_resized,
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(
            frame,
            0.65,
            heatmap_colored,
            0.35,
            0,
        )

        cv2.imwrite(str(overlay_output_path), overlay)

        print("\nHeatmap generation complete.")
        print(f"Saved heatmap: {heatmap_output_path}")
        print(f"Saved overlay: {overlay_output_path}")

        return {
            "total_points": len(tracking_df),
            "filtered_points": len(filtered_df),
            "valid_player_ids": len(valid_track_ids),
            "heatmap_path": str(heatmap_output_path),
            "overlay_path": str(overlay_output_path),
        }

    def _get_first_frame(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Could not open video.")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read first video frame.")

        return frame

    def _resize_frame(self, frame):
        original_height, original_width = frame.shape[:2]

        if self.resize_width is None or original_width <= self.resize_width:
            return frame

        scale = self.resize_width / original_width
        new_height = int(original_height * scale)

        return cv2.resize(frame, (self.resize_width, new_height))
