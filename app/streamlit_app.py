from pathlib import Path
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Sports Player Tracking Dashboard",
    page_icon="⚽",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parents[1]

TRACKING_CSV = BASE_DIR / "data/tracking_data/player_tracking.csv"
MOVEMENT_SUMMARY_CSV = BASE_DIR / "data/tracking_data/player_movement_summary.csv"
PERFORMANCE_SUMMARY_CSV = BASE_DIR / "data/tracking_data/player_performance_summary.csv"

TRACKED_VIDEO = BASE_DIR / "data/output_videos/tracked_match_clip.mp4"
TRAIL_VIDEO = BASE_DIR / "data/output_videos/tracked_with_trails.mp4"
HEATMAP = BASE_DIR / "data/output_videos/player_heatmap.png"
HEATMAP_OVERLAY = BASE_DIR / "data/output_videos/player_heatmap_overlay.png"


@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def show_file_warning(path, label):
    if not path.exists():
        st.warning(f"Missing file: {label} → {path}")


def main():
    st.title("⚽ AI Sports Player Tracking & Movement Analysis")
    st.write(
        "Computer vision dashboard for player detection, tracking, movement trails, "
        "speed analysis, distance covered, and heatmap visualization."
    )

    tracking_df = load_csv(TRACKING_CSV)
    movement_df = load_csv(MOVEMENT_SUMMARY_CSV)
    performance_df = load_csv(PERFORMANCE_SUMMARY_CSV)

    if tracking_df.empty:
        st.error("Tracking data not found. Run `python main.py` first.")
        return

    st.sidebar.title("Dashboard Controls")

    min_frames = st.sidebar.slider(
        "Minimum frames seen",
        min_value=1,
        max_value=int(performance_df["frames_seen"].max()) if not performance_df.empty else 100,
        value=30,
    )

    filtered_perf = performance_df.copy()

    if not filtered_perf.empty:
        filtered_perf = filtered_perf[filtered_perf["frames_seen"] >= min_frames]

    st.header("1. Project Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Detections", f"{len(tracking_df):,}")
    col2.metric("Raw Player IDs", tracking_df["track_id"].nunique())
    col3.metric("Stable Player IDs", len(filtered_perf))
    col4.metric("Frames Processed", tracking_df["frame"].nunique())

    st.divider()

    st.header("2. Output Videos")

    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.subheader("Tracked Video")
        show_file_warning(TRACKED_VIDEO, "tracked_match_clip.mp4")
        if TRACKED_VIDEO.exists():
            st.video(str(TRACKED_VIDEO))

    with video_col2:
        st.subheader("Movement Trail Video")
        show_file_warning(TRAIL_VIDEO, "tracked_with_trails.mp4")
        if TRAIL_VIDEO.exists():
            st.video(str(TRAIL_VIDEO))

    st.divider()

    st.header("3. Player Performance Summary")

    if filtered_perf.empty:
        st.warning("No players found after filtering. Lower the minimum frames slider.")
    else:
        st.dataframe(
            filtered_perf.sort_values("total_distance_pixels", ascending=False),
            use_container_width=True,
        )

        top_distance = filtered_perf.sort_values(
            "total_distance_pixels",
            ascending=False,
        ).head(10)

        st.subheader("Top Players by Distance Covered")
        st.bar_chart(
            top_distance.set_index("track_id")["total_distance_pixels"]
        )

        top_speed = filtered_perf.sort_values(
            "avg_speed_pixels_per_second",
            ascending=False,
        ).head(10)

        st.subheader("Top Players by Average Speed")
        st.bar_chart(
            top_speed.set_index("track_id")["avg_speed_pixels_per_second"]
        )

    st.divider()

    st.header("4. Movement Heatmaps")

    heatmap_col1, heatmap_col2 = st.columns(2)

    with heatmap_col1:
        st.subheader("Standalone Heatmap")
        show_file_warning(HEATMAP, "player_heatmap.png")
        if HEATMAP.exists():
            st.image(str(HEATMAP), use_container_width=True)

    with heatmap_col2:
        st.subheader("Heatmap Overlay")
        show_file_warning(HEATMAP_OVERLAY, "player_heatmap_overlay.png")
        if HEATMAP_OVERLAY.exists():
            st.image(str(HEATMAP_OVERLAY), use_container_width=True)

    st.divider()

    st.header("5. Raw Tracking Data")

    with st.expander("View raw tracking data"):
        st.dataframe(tracking_df.head(1000), use_container_width=True)

    with st.expander("View movement summary"):
        if movement_df.empty:
            st.warning("Movement summary file missing.")
        else:
            st.dataframe(movement_df, use_container_width=True)

    st.divider()

    st.header("6. Key Insights")

    if not filtered_perf.empty:
        best_distance = filtered_perf.sort_values(
            "total_distance_pixels",
            ascending=False,
        ).iloc[0]

        best_speed = filtered_perf.sort_values(
            "avg_speed_pixels_per_second",
            ascending=False,
        ).iloc[0]

        st.success(
            f"Player {int(best_distance['track_id'])} covered the highest tracked distance: "
            f"{best_distance['total_distance_pixels']:.2f} pixels."
        )

        st.info(
            f"Player {int(best_speed['track_id'])} had the highest average tracked speed: "
            f"{best_speed['avg_speed_pixels_per_second']:.2f} pixels/second."
        )

        st.warning(
            "Current distance and speed are pixel-based. For real-world meters/second, "
            "field calibration or homography is required."
        )


if __name__ == "__main__":
    main()
