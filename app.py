import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import tempfile
import os

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is not None and points.size > 0:
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
            return transformed_points.reshape(-1, 2)
        else:
            return None

def process_video(video_path, progress_bar):
    video_info = sv.VideoInfo.from_video_path(video_path)
    model = YOLO("best.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_thickness=thickness)

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, video_info.resolution_wh)

    # Get the total number of frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_generator = sv.get_video_frames_generator(video_path)
    frame_idx = 0

    for frame in frame_generator:
        result = model(frame)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points)

        if points is not None:
            points = points.astype(int)

            labels = []
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")
                    print(f"Tracker ID: {tracker_id}, Speed: {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.polylines(annotated_frame, [SOURCE], isClosed=True, color=(0, 0, 255), thickness=thickness)

            out.write(annotated_frame)
        else:
            print("No points detected in this frame.")

        frame_idx += 1
        progress_percentage = int((frame_idx / total_frames) * 100)
        progress_bar.progress(progress_percentage, text="Processing frames...")

    out.release()

    if os.path.exists(output_path):
        return output_path
    else:
        st.error("Error: Output file was not created.")
        return None

def main():
    st.title("Vehicle Detection and Speed Estimation")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        if st.button('Start'):
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name

            # Initialize the progress bar
            progress_bar = st.progress(0, text="Processing frames...")

            # Process the video and update the progress bar
            output_path = process_video(video_path, progress_bar)

            if output_path:
                st.success("Processing complete!")
                with open(output_path, 'rb') as f:
                    st.download_button('Download Processed Video', f, file_name='processed_video.avi')
            else:
                st.error("Error processing the video.")
    else:
        st.error("Please upload a video file.")

if __name__ == "__main__":
    main()
