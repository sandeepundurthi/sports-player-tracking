from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


class PlayerTracker:
    def __init__(
        self,
        model_name="yolov8n.pt",
        tracker_type="bytetrack.yaml",
        conf_threshold=0.35,
        resize_width=1280,
    ):
        """
        Initializes YOLO player tracker.

        model_name:
            YOLO model file. yolov8n.pt is fastest.
            Later we can try yolov8s.pt for better accuracy.

        tracker_type:
            bytetrack.yaml or botsort.yaml.

        conf_threshold:
            Minimum confidence score for detections.

        resize_width:
            Your video is 4K, so we resize to 1280 width for faster processing.
        """

        self.model = YOLO(model_name)
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.resize_width = resize_width

    def resize_frame(self, frame):
        """
        Resize frame while keeping aspect ratio.
        """

        original_height, original_width = frame.shape[:2]

        if self.resize_width is None or original_width <= self.resize_width:
            return frame

        scale = self.resize_width / original_width
        new_height = int(original_height * scale)

        resized_frame = cv2.resize(frame, (self.resize_width, new_height))
        return resized_frame

    def process_video(self, input_video_path, output_video_path, csv_output_path):
        """
        Runs YOLO tracking on the input video.

        Saves:
        1. Annotated output video
        2. Tracking CSV file
        """

        input_video_path = Path(input_video_path)
        output_video_path = Path(output_video_path)
        csv_output_path = Path(csv_output_path)

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(str(input_video_path))

        if not cap.isOpened():
            raise ValueError("Could not open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame.")

        first_frame = self.resize_frame(first_frame)
        output_height, output_width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (output_width, output_height),
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        tracking_records = []

        print("Starting player tracking...")
        print(f"Input video: {input_video_path}")
        print(f"Output video: {output_video_path}")
        print(f"Tracking CSV: {csv_output_path}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Output size: {output_width}x{output_height}")

        for frame_index in tqdm(range(total_frames)):
            ret, frame = cap.read()

            if not ret:
                break

            frame = self.resize_frame(frame)

            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_type,
                conf=self.conf_threshold,
                classes=[0],
                verbose=False,
            )

            annotated_frame = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                if boxes.id is not None:
                    for box in boxes:
                        track_id = int(box.id.item())
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())

                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        tracking_records.append(
                            {
                                "frame": frame_index,
                                "time_seconds": frame_index / fps,
                                "track_id": track_id,
                                "class_id": class_id,
                                "confidence": confidence,
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "center_x": center_x,
                                "center_y": center_y,
                                "bbox_width": width,
                                "bbox_height": height,
                            }
                        )

                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )

                        label = f"Player {track_id} | {confidence:.2f}"

                        cv2.putText(
                            annotated_frame,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                        cv2.circle(
                            annotated_frame,
                            (center_x, center_y),
                            4,
                            (0, 0, 255),
                            -1,
                        )

            writer.write(annotated_frame)

        cap.release()
        writer.release()

        tracking_df = pd.DataFrame(tracking_records)
        tracking_df.to_csv(csv_output_path, index=False)

        print("\nTracking complete.")
        print(f"Saved annotated video to: {output_video_path}")
        print(f"Saved tracking data to: {csv_output_path}")

        if not tracking_df.empty:
            print("\nTracking Summary")
            print("----------------")
            print(f"Total detections: {len(tracking_df)}")
            print(f"Unique players tracked: {tracking_df['track_id'].nunique()}")
            print(f"Frames with detections: {tracking_df['frame'].nunique()}")
        else:
            print("No players detected. Try lowering conf_threshold to 0.25.")

        return tracking_df
