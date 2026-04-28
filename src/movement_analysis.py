from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque


class MovementTrailAnalyzer:
    def __init__(self, trail_length=25):
        self.trail_length = trail_length

    def create_trail_video(
        self,
        input_video_path,
        tracking_csv_path,
        output_video_path,
        resize_width=1280,
    ):
        input_video_path = Path(input_video_path)
        tracking_csv_path = Path(tracking_csv_path)
        output_video_path = Path(output_video_path)

        if not input_video_path.exists():
            raise FileNotFoundError(f"Video not found: {input_video_path}")

        if not tracking_csv_path.exists():
            raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        tracking_df = pd.read_csv(tracking_csv_path)

        if tracking_df.empty:
            raise ValueError("Tracking CSV is empty.")

        frame_groups = {
            frame: group
            for frame, group in tracking_df.groupby("frame")
        }

        cap = cv2.VideoCapture(str(input_video_path))

        if not cap.isOpened():
            raise ValueError("Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame.")

        first_frame = self._resize_frame(first_frame, resize_width)
        output_height, output_width = first_frame.shape[:2]

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        player_trails = defaultdict(lambda: deque(maxlen=self.trail_length))

        print("Creating movement trail video...")

        for frame_index in tqdm(range(total_frames)):
            ret, frame = cap.read()

            if not ret:
                break

            frame = self._resize_frame(frame, resize_width)

            if frame_index in frame_groups:
                current_frame_data = frame_groups[frame_index]

                for _, row in current_frame_data.iterrows():
                    track_id = int(row["track_id"])
                    center_x = int(row["center_x"])
                    center_y = int(row["center_y"])

                    player_trails[track_id].append((center_x, center_y))

                    x1 = int(row["x1"])
                    y1 = int(row["y1"])
                    x2 = int(row["x2"])
                    y2 = int(row["y2"])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"Player {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            for track_id, trail_points in player_trails.items():
                points = list(trail_points)

                for i in range(1, len(points)):
                    thickness = int(2 + (i / len(points)) * 4)
                    cv2.line(frame, points[i - 1], points[i], (255, 0, 0), thickness)

                if len(points) > 0:
                    cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)

            writer.write(frame)

        cap.release()
        writer.release()

        print("\nTrail video complete.")
        print(f"Saved to: {output_video_path}")

    def create_movement_summary(self, tracking_csv_path, output_csv_path):
        tracking_csv_path = Path(tracking_csv_path)
        output_csv_path = Path(output_csv_path)

        df = pd.read_csv(tracking_csv_path)

        if df.empty:
            raise ValueError("Tracking CSV is empty.")

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        summary_rows = []

        for track_id, group in df.groupby("track_id"):
            group = group.sort_values("frame")

            summary_rows.append(
                {
                    "track_id": int(track_id),
                    "frames_seen": group["frame"].nunique(),
                    "first_frame": int(group["frame"].min()),
                    "last_frame": int(group["frame"].max()),
                    "first_time_seconds": round(float(group["time_seconds"].min()), 2),
                    "last_time_seconds": round(float(group["time_seconds"].max()), 2),
                    "duration_tracked_seconds": round(
                        float(group["time_seconds"].max() - group["time_seconds"].min()), 2
                    ),
                    "avg_confidence": round(float(group["confidence"].mean()), 3),
                    "start_x": int(group.iloc[0]["center_x"]),
                    "start_y": int(group.iloc[0]["center_y"]),
                    "end_x": int(group.iloc[-1]["center_x"]),
                    "end_y": int(group.iloc[-1]["center_y"]),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by="frames_seen", ascending=False)
        summary_df.to_csv(output_csv_path, index=False)

        print("\nMovement summary saved.")
        print(f"Saved to: {output_csv_path}")
        print("\nTop tracked players:")
        print(summary_df.head(10))

        return summary_df

    def calculate_speed_and_distance(
        self,
        tracking_csv_path,
        detailed_output_csv_path,
        summary_output_csv_path,
        fps=25.0,
    ):
        """
        Calculates player speed and distance using player center points.

        Current unit:
        - distance = pixels
        - speed = pixels per second

        Later, we can convert pixels to meters using field calibration.
        """

        tracking_csv_path = Path(tracking_csv_path)
        detailed_output_csv_path = Path(detailed_output_csv_path)
        summary_output_csv_path = Path(summary_output_csv_path)

        if not tracking_csv_path.exists():
            raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")

        detailed_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(tracking_csv_path)

        if df.empty:
            raise ValueError("Tracking CSV is empty.")

        all_rows = []
        summary_rows = []

        print("\nCalculating speed and distance...")

        for track_id, group in df.groupby("track_id"):
            group = group.sort_values("frame").reset_index(drop=True)

            previous_x = None
            previous_y = None
            previous_frame = None

            total_distance_pixels = 0.0
            speeds = []

            for _, row in group.iterrows():
                current_x = float(row["center_x"])
                current_y = float(row["center_y"])
                current_frame = int(row["frame"])

                if previous_x is None:
                    frame_distance = 0.0
                    frame_gap = 0
                    time_gap_seconds = 0.0
                    speed_pixels_per_second = 0.0
                else:
                    frame_gap = current_frame - previous_frame

                    if frame_gap <= 0:
                        frame_distance = 0.0
                        time_gap_seconds = 0.0
                        speed_pixels_per_second = 0.0
                    else:
                        dx = current_x - previous_x
                        dy = current_y - previous_y

                        frame_distance = np.sqrt(dx**2 + dy**2)
                        time_gap_seconds = frame_gap / fps
                        speed_pixels_per_second = frame_distance / time_gap_seconds

                    total_distance_pixels += frame_distance
                    speeds.append(speed_pixels_per_second)

                all_rows.append(
                    {
                        "frame": current_frame,
                        "time_seconds": row["time_seconds"],
                        "track_id": int(track_id),
                        "center_x": int(current_x),
                        "center_y": int(current_y),
                        "frame_gap": frame_gap,
                        "distance_from_previous_pixels": round(frame_distance, 3),
                        "speed_pixels_per_second": round(speed_pixels_per_second, 3),
                        "cumulative_distance_pixels": round(total_distance_pixels, 3),
                    }
                )

                previous_x = current_x
                previous_y = current_y
                previous_frame = current_frame

            valid_speeds = [s for s in speeds if s > 0]

            summary_rows.append(
                {
                    "track_id": int(track_id),
                    "frames_seen": int(group["frame"].nunique()),
                    "duration_seconds": round(
                        float(group["time_seconds"].max() - group["time_seconds"].min()), 2
                    ),
                    "total_distance_pixels": round(total_distance_pixels, 2),
                    "avg_speed_pixels_per_second": round(
                        float(np.mean(valid_speeds)) if valid_speeds else 0.0, 2
                    ),
                    "max_speed_pixels_per_second": round(
                        float(np.max(valid_speeds)) if valid_speeds else 0.0, 2
                    ),
                    "avg_confidence": round(float(group["confidence"].mean()), 3),
                }
            )

        detailed_df = pd.DataFrame(all_rows)
        summary_df = pd.DataFrame(summary_rows)

        summary_df = summary_df.sort_values(
            by="total_distance_pixels",
            ascending=False,
        )

        detailed_df.to_csv(detailed_output_csv_path, index=False)
        summary_df.to_csv(summary_output_csv_path, index=False)

        print("\nSpeed and distance calculation complete.")
        print(f"Detailed file saved to: {detailed_output_csv_path}")
        print(f"Summary file saved to: {summary_output_csv_path}")

        print("\nTop players by distance covered:")
        print(summary_df.head(10))

        return detailed_df, summary_df

    def _resize_frame(self, frame, resize_width):
        original_height, original_width = frame.shape[:2]

        if resize_width is None or original_width <= resize_width:
            return frame

        scale = resize_width / original_width
        new_height = int(original_height * scale)

        return cv2.resize(frame, (resize_width, new_height))
