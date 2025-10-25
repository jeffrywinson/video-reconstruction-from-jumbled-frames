import os
import sys
import time
# We no longer need tqdm here, it's moved to feature_extraction.py
# from tqdm import tqdm 

from src.video_processing import extract_frames, save_video
from src.feature_extraction import FeatureExtractor
from src.solving import build_distance_matrix, solve_tsp

# --- Configuration ---
INPUT_DIR = 'test_videos'
OUTPUT_DIR = 'output_videos'
VIDEO_NAME = 'jumbled_video.mp4'  # The video file to process
# ---------------------

def run_reconstruction():
    
    input_path = os.path.join(INPUT_DIR, VIDEO_NAME)
    output_path = os.path.join(OUTPUT_DIR, f"reconstructed_{VIDEO_NAME}")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please make sure 'jumbled_video.mp4' is in the 'test_videos' folder.")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start_time = time.time()

    # --- Step 1: Extract Frames ---
    print(f"Loading and extracting frames from {input_path}...")
    frames, fps, resolution = extract_frames(input_path)
    if frames is None:
        return
    
    num_frames = len(frames)
    print(f"Successfully extracted {num_frames} frames ({resolution[0]}x{resolution[1]} @ {fps:.2f} FPS).")

    # --- Step 2: Extract Features ---
    print("Initializing feature extractor (loading ResNet-50)...")
    extractor = FeatureExtractor()
    print(f"Feature extractor is using device: {extractor.device}")
    
    print("Extracting features from all frames (in a single batch)...")
    
    # --- THIS IS THE MODIFIED PART ---
    start_feature_time = time.time()
    
    # We now call our new batch function once.
    features_list = extractor.extract_batch(frames)
    
    print(f"Batch feature extraction finished in {time.time() - start_feature_time:.2f}s.")
    # --- END OF MODIFICATION ---
    
    print(f"Extracted {len(features_list)} feature vectors.")

    # --- Step 3: Build Distance Matrix ---
    print("Building N x N distance matrix...")
    start_matrix_time = time.time()
    distance_matrix = build_distance_matrix(features_list)
    print(f"Matrix built in {time.time() - start_matrix_time:.2f}s.")
    
    # --- Step 4: Solve TSP ---
    print("Solving for the optimal path (TSP)...")
    start_solve_time = time.time()
    solved_order = solve_tsp(distance_matrix)
    print(f"Path solved in {time.time() - start_solve_time:.2f}s.")
    
    if len(solved_order) != num_frames:
        print(f"Error: Solved path length ({len(solved_order)}) does not match frame count ({num_frames}).")
        return

    # --- Step 5: Save Reconstructed Video ---
    print(f"Saving reconstructed video to {output_path}...")
    save_video(frames, solved_order, output_path, fps, resolution)
    
    total_end_time = time.time()
    print("\n--- Reconstruction Complete! ---")
    print(f"Output saved to: {output_path}")
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds.")


if __name__ == "__main__":
    run_reconstruction()