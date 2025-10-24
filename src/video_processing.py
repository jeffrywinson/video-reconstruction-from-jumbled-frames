import cv2
import numpy as np

def extract_frames(video_path):
    """
    Reads a video file and extracts all frames into a list.
    Also returns the original FPS and frame dimensions.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, resolution

def save_video(frames, frame_order, output_path, fps, resolution):
    """
    Saves a list of frames into a new video file, based on the solved order.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    if not writer.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return

    for frame_index in frame_order:
        writer.write(frames[frame_index])
    
    writer.release()