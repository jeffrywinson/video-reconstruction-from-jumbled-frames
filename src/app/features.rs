use anyhow::{anyhow, Result};
use image::{imageops::FilterType, RgbImage};
use ndarray::{Array, Array2, Array3, Array4, ArrayD, Axis, Ix2};
use opencv::prelude::*;
use ort::{
    self,
    session::{builder::GraphOptimizationLevel, Session},
    value::{Value, Tensor},
};

// ImageNet normalization constants
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct FeatureExtractor {
    model_path: String,
}

impl FeatureExtractor {
    pub fn new(model_path: &str) -> Result<Self> {
        // Initialize global ORT environment (do once)
        ort::init().with_name("FrameExtractor").commit()?;
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }

    /// Preprocess a single OpenCV Mat into Array3<f32> (C, H, W)
    fn preprocess_frame(&self, frame: &Mat) -> Result<Array3<f32>> {
        let cols = frame.cols() as u32;
        let rows = frame.rows() as u32;
        let data = frame.data_bytes()?;

        // Convert BGR -> RGB
        let mut rgb_image = RgbImage::new(cols, rows);
        for y in 0..rows {
            for x in 0..cols {
                let i = (y * cols + x) as usize * 3;
                rgb_image.put_pixel(x, y, image::Rgb([data[i + 2], data[i + 1], data[i]]));
            }
        }

        // Resize and normalize
        let resized = image::imageops::resize(&rgb_image, 224, 224, FilterType::Triangle);
        let mut arr = Array::zeros((3, 224, 224)); // C, H, W
        for (x, y, pixel) in resized.enumerate_pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            arr[[0, y as usize, x as usize]] = (r - MEAN[0]) / STD[0];
            arr[[1, y as usize, x as usize]] = (g - MEAN[1]) / STD[1];
            arr[[2, y as usize, x as usize]] = (b - MEAN[2]) / STD[2];
        }
        Ok(arr)
    }

    /// Run a single batch inference over many frames. Returns (batch_size, feature_dim) array.
    pub fn extract_batch(&self, frames: &[Mat]) -> Result<Array<f32, Ix2>> {
        // Build session (session.run expects &mut self)
        let mut session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&self.model_path)?;

        println!("Pre-processing {} frames...", frames.len());

        // Preprocess frames
        let tensors_res: Result<Vec<Array3<f32>>> =
            frames.iter().map(|f| self.preprocess_frame(f)).collect();
        let tensors = tensors_res?;

        // Stack into Array4: (batch, C, H, W)
        let batch_views: Vec<_> = tensors.iter().map(|t| t.view()).collect();
        let batch: Array4<f32> = ndarray::stack(Axis(0), &batch_views)?;

        // Convert to ArrayD and get raw vec + shape
        let batch_dyn: ArrayD<f32> = batch.into_dyn();
        let shape_usize: Vec<usize> = batch_dyn.shape().to_vec();
        let data_vec: Vec<f32> = batch_dyn.into_raw_vec();

        // Build ORT Tensor from owned shape + data
        let input_tensor: Tensor<f32> = Tensor::from_array((shape_usize.as_slice(), data_vec))?;

        println!("Running model inference...");
        // Build inputs vector explicitly typed to avoid ambiguity on .into()
        let inputs: Vec<(&str, Value)> = vec![("input", input_tensor.into())];

        // Run session (session.run takes &mut self)
        let outputs_collection = session.run(inputs)?;

        // Extract first output (name, Value)
        let (_, features_value): (&str, Value) = outputs_collection
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Model did not return an output tensor"))?;

        // On RC.7 try_extract_tensor::<T>() returns (&Shape, &[T])
        let (shape, data_slice): (&ort::tensor::Shape, &[f32]) =
            features_value.try_extract_tensor::<f32>()?;

        // Obtain dims as &[i64] via as_ref() (works on RC.7)
        let dims_i64: &[i64] = shape.as_ref();
        if dims_i64.len() < 2 {
            return Err(anyhow!(
                "Unexpected output tensor rank from model: {:?}",
                dims_i64
            ));
        }

        // Convert dims to usize vector
        let shape_vec: Vec<usize> = dims_i64.iter().map(|&d| d as usize).collect();

        // Handle expected output shapes:
        //  - [batch, features, 1, 1] -> flatten to (batch, features)
        //  - [batch, features] -> use directly
        let features_matrix: Array2<f32> = match shape_vec.as_slice() {
            [batch, features, 1, 1] => {
                Array::from_shape_vec((*batch, *features), data_slice.to_vec())?
            }
            [batch, features] => Array::from_shape_vec((*batch, *features), data_slice.to_vec())?,
            other => {
                return Err(anyhow!(
                    "Unhandled output tensor shape from model: {:?}. Expected [batch,features,1,1] or [batch,features].",
                    other
                ));
            }
        };

        Ok(features_matrix)
    }
}
