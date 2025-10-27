use anyhow::{anyhow, Result};
use opencv::{
    core::{Mat, Size_},
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};

/// Reads a video file and extracts all frames into a Vec.
/// Also returns the original FPS and frame dimensions.
pub fn extract_frames(video_path: &str) -> Result<(Vec<Mat>, f64, Size_<i32>)> {
    let mut cap = VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    
    if !cap.is_opened()? {
        return Err(anyhow!("Error: Could not open video file {}", video_path));
    }

    let mut frames = Vec::new();
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let resolution = Size_::new(width, height);

    let mut frame = Mat::default();
    while cap.read(&mut frame)? {
        if frame.empty() {
            break;
        }
        // Push a *clone* of the frame into our vector
        frames.push(frame.clone());
    }
    
    cap.release()?;
    Ok((frames, fps, resolution))
}

/// Saves a list of frames into a new video file, based on the solved order.
pub fn save_video(
    frames: &[Mat],
    frame_order: &[usize],
    output_path: &str,
    fps: f64,
    resolution: Size_<i32>,
) -> Result<()> {
    // Define the codec
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = VideoWriter::new(output_path, fourcc, fps, resolution, true)?;

    if !writer.is_opened()? {
        return Err(anyhow!("Error: Could not create video writer for {}", output_path));
    }

    // Write frames in the new order
    for &frame_index in frame_order {
        writer.write(&frames[frame_index])?;
    }
    
    writer.release()?;
    Ok(())
}