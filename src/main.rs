use anyhow::Result; // Easy error handling
use std::time::Instant;

// Tell Rust to look for the 'app' module folder
mod app;

// --- DEFINITIVE FIX: Use Absolute Paths Inside Container ---
const INPUT_DIR: &str = "/app/test_videos"; // Absolute path
const OUTPUT_DIR: &str = "/app/output_videos"; // Absolute path
const VIDEO_NAME: &str = "jumbled_video.mp4";
const MODEL_PATH: &str = "resnet50.onnx"; 
// -----------------------------------------------------------

fn main() -> Result<()> {
    // Ensure output directory exists (using absolute path)
    std::fs::create_dir_all(OUTPUT_DIR)?;

    let input_path = format!("{}/{}", INPUT_DIR, VIDEO_NAME);
    let output_path = format!("{}/reconstructed_{}", OUTPUT_DIR, VIDEO_NAME);

    let total_start = Instant::now();

    // --- Step 1: Extract Frames ---
    println!("Loading and extracting frames from {}...", input_path);
    let (frames, fps, resolution) = app::video_io::extract_frames(&input_path)?;
    let num_frames = frames.len();
    println!(
        "Successfully extracted {} frames ({}x{} @ {:.2} FPS).",
        num_frames, resolution.width, resolution.height, fps
    );

    // --- Step 2: Extract Features ---
    println!("Initializing feature extractor (loading ResNet-50)...");
    // --- DEFINITIVE FIX: Use absolute path for model ---
    let extractor = app::features::FeatureExtractor::new(MODEL_PATH)?;

    println!("Extracting features from all frames (sequentially)..."); // Update message if not parallel
    let feature_start = Instant::now();

    let features = extractor.extract_batch(&frames)?;

    println!(
        "Batch feature extraction finished in {:.2}s.",
        feature_start.elapsed().as_secs_f32()
    );
    println!("Extracted {} feature vectors.", features.shape()[0]);

    // --- Step 3: Build Distance Matrix ---
    println!("Building N x N distance matrix...");
    let matrix_start = Instant::now();

    let distance_matrix = app::solver::build_distance_matrix(features.view())?;

    println!(
        "Matrix built in {:.2}s.",
        matrix_start.elapsed().as_secs_f32()
    );

    // --- Step 4: Solve TSP ---
    println!("Solving for the optimal path (Native LKH-3)...");
    let solve_start = Instant::now();

    let solved_order = app::solver::solve_tsp(&distance_matrix)?;

    println!(
        "Path solved in {:.2}s.",
        solve_start.elapsed().as_secs_f32()
    );

    if solved_order.len() != num_frames {
        eprintln!("Error: Solved path length does not match frame count.");
        // Consider returning an error instead of just printing
        return Err(anyhow::anyhow!("Solved path length mismatch"));
    }

    // --- Step 5: Save Reconstructed Video ---
    println!("Saving reconstructed video to {}...", output_path);
    app::video_io::save_video(&frames, &solved_order, &output_path, fps, resolution)?;

    println!("\n--- Reconstruction Complete! ---");
    println!("Output saved to: {}", output_path);
    println!(
        "Total processing time: {:.2} seconds.",
        total_start.elapsed().as_secs_f32()
    );

    Ok(())
}