from src.video_info import get_video_info
from src.player_tracker import PlayerTracker
from src.movement_analysis import MovementTrailAnalyzer
from src.heatmap_generator import PlayerHeatmapGenerator


VIDEO_PATH = "data/input_videos/match_clip.mp4"

TRACKED_VIDEO_PATH = "data/output_videos/tracked_match_clip.mp4"
TRAIL_VIDEO_PATH = "data/output_videos/tracked_with_trails.mp4"

TRACKING_CSV_PATH = "data/tracking_data/player_tracking.csv"
MOVEMENT_SUMMARY_CSV_PATH = "data/tracking_data/player_movement_summary.csv"

SPEED_DISTANCE_CSV_PATH = "data/tracking_data/player_speed_distance.csv"
PERFORMANCE_SUMMARY_CSV_PATH = "data/tracking_data/player_performance_summary.csv"

HEATMAP_PATH = "data/output_videos/player_heatmap.png"
HEATMAP_OVERLAY_PATH = "data/output_videos/player_heatmap_overlay.png"


def main():
    print("Checking video information...")
    info = get_video_info(VIDEO_PATH)

    print("\nVideo Information")
    print("-----------------")
    for key, value in info.items():
        print(f"{key}: {value}")

    fps = info["fps"]

    print("\nRunning YOLO player tracking...")

    tracker = PlayerTracker(
        model_name="yolov8n.pt",
        tracker_type="bytetrack.yaml",
        conf_threshold=0.35,
        resize_width=1280,
    )

    tracker.process_video(
        input_video_path=VIDEO_PATH,
        output_video_path=TRACKED_VIDEO_PATH,
        csv_output_path=TRACKING_CSV_PATH,
    )

    print("\nCreating movement trail analysis...")

    analyzer = MovementTrailAnalyzer(trail_length=25)

    analyzer.create_trail_video(
        input_video_path=VIDEO_PATH,
        tracking_csv_path=TRACKING_CSV_PATH,
        output_video_path=TRAIL_VIDEO_PATH,
        resize_width=1280,
    )

    analyzer.create_movement_summary(
        tracking_csv_path=TRACKING_CSV_PATH,
        output_csv_path=MOVEMENT_SUMMARY_CSV_PATH,
    )

    analyzer.calculate_speed_and_distance(
        tracking_csv_path=TRACKING_CSV_PATH,
        detailed_output_csv_path=SPEED_DISTANCE_CSV_PATH,
        summary_output_csv_path=PERFORMANCE_SUMMARY_CSV_PATH,
        fps=fps,
    )

    print("\nGenerating player heatmap...")

    heatmap_generator = PlayerHeatmapGenerator(resize_width=1280)

    heatmap_summary = heatmap_generator.generate_heatmap(
        video_path=VIDEO_PATH,
        tracking_csv_path=TRACKING_CSV_PATH,
        heatmap_output_path=HEATMAP_PATH,
        overlay_output_path=HEATMAP_OVERLAY_PATH,
        min_frames_seen=30,
        bins=80,
    )

    print("\nHeatmap Summary")
    print("---------------")
    for key, value in heatmap_summary.items():
        print(f"{key}: {value}")

    print("\nStep 7 complete.")
    print("Generated files:")
    print(f"- {HEATMAP_PATH}")
    print(f"- {HEATMAP_OVERLAY_PATH}")


if __name__ == "__main__":
    main()
