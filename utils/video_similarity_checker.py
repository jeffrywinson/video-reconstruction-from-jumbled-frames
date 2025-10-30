import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_videos(video_path_1, video_path_2):
    """
    Compares two videos frame by frame using SSIM and calculates the
    average similarity.
    """
    
    # Open video capture objects
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    # Check if videos opened successfully
    if not cap1.isOpened():
        print(f"Error: Could not open video 1 at {video_path_1}")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open video 2 at {video_path_2}")
        return

    frame_similarities = []
    frame_count = 0

    print("--- Starting Frame-by-Frame Comparison ---")

    while True:
        # Read a frame from each video
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # If either video ends, stop the comparison
        if not ret1 or not ret2:
            break

        frame_count += 1

        # --- Pre-processing for SSIM ---
        
        # 1. Check for matching dimensions. Resize if they don't match.
        if frame1.shape != frame2.shape:
            print(f"Warning: Frame {frame_count} dimensions differ. Resizing frame 2 to match frame 1.")
            height, width, _ = frame1.shape
            frame2 = cv2.resize(frame2, (width, height))

        # 2. Convert frames to grayscale. SSIM is typically run on single-channel images.
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 3. Calculate SSIM
        # The 'data_range' is the dynamic range of the image (e.g., 255 for 8-bit images)
        score = ssim(gray1, gray2, data_range=gray1.max() - gray1.min())
        
        frame_similarities.append(score)
        print(f"Frame {frame_count}: Similarity Score = {score:.4f}")

    # Release the video capture objects
    cap1.release()
    cap2.release()

    print("--- Comparison Finished ---")

    # --- Calculate and Print Results ---
    if not frame_similarities:
        print("No frames were compared.")
        return

    # Calculate the average similarity
    # SSIM scores are in the range [-1, 1], where 1 is perfect similarity.
    average_similarity = np.mean(frame_similarities)
    
    # Convert to a percentage (assuming 1.0 = 100%)
    average_similarity_percent = average_similarity * 100

    print(f"\n--- Final Report ---")
    print(f"Total frames compared: {frame_count}")
    print(f"Average Similarity: {average_similarity_percent:.2f}%")


# --- Main execution ---
if __name__ == "__main__":
    # Get paths from user
    dir1 = input("Enter path to directory 1: ")
    file1 = input("Enter video file name in directory 1 (e.g., vid1.mp4): ")
    dir2 = input("Enter path to directory 2: ")
    file2 = input("Enter video file name in directory 2 (e.g., vid2.mp4): ")

    # Construct full paths
    vid_path1 = os.path.join(dir1, file1)
    vid_path2 = os.path.join(dir2, file2)

    # Check if files exist
    if not os.path.exists(vid_path1):
        print(f"Error: File not found at {vid_path1}")
    elif not os.path.exists(vid_path2):
        print(f"Error: File not found at {vid_path2}")
    else:
        compare_videos(vid_path1, vid_path2)