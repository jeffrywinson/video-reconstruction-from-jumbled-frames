# Jumbled Frames Reconstruction Challenge (High-Performance Rust Solution) 🚀

This project reconstructs a 10-second, 30 FPS video from 300 randomly shuffled frames using a high-performance, native Rust application packaged in Docker.

The core algorithm leverages Convolutional Neural Networks (CNN) for semantic frame analysis and solves the frame ordering problem as a Traveling Salesperson Problem (TSP) using the highly optimized LKH-3 heuristic.

**Key Technologies:**
* **Language:** Rust 🦀
* **AI Model:** ResNet-50 (via ONNX Runtime / `ort` crate)
* **Video I/O:** OpenCV (`opencv-rust` crate) 📷
* **TSP Solver:** LKH-3 heuristic (`elkai-rs` crate) 🗺️
* **Containerization:** Docker 🐳

---

## ⚡ Quick Start: Running the Pre-Built Solution

The entire application, including all complex C++/OpenCV/ONNX dependencies, has been pre-compiled into a Docker image and hosted on the GitHub Container Registry (GHCR).

**Prerequisite:**
* **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** installed and running.
    * *No Rust, Python, C++, CMake, or OpenCV installation is required on your machine.*

**Steps:**

1.  **Clone the Repository (Optional but Recommended):**
    This gives you the necessary `test_videos` and `output_videos` folder structure.
    ```bash
    git clone https://github.com/jeffrywinson/video-reconstruction-from-jumbled-frames
    cd video-reconstruction-from-jumbled-frames
    # Create output directory if it doesn't exist
    mkdir -p output_videos
    ```

2.  **Prepare Input:**
    Place your jumbled video file (e.g., `jumbled_video.mp4`) inside the `test_videos` directory.

3.  **Run the Docker Container:**
    Open your terminal in the project's root directory (`video-reconstruction-from-jumbled-frames`) and execute the command below.
    * This command automatically **downloads** the pre-built Docker image (~1-2 GB download the first time).
    * It then **instantly runs** the compiled Rust program (< 15 seconds runtime expected on the benchmark i7).
    * It mounts your local `test_videos` and `output_videos` folders for input/output.

    ```bash
    # Make sure Docker Desktop is running!
    # Replace ghcr.io/... path with your actual image path if different
    docker run --rm \
      -v "//$(pwd)/test_videos":/app/test_videos \
      -v "//$(pwd)/output_videos":/app/output_videos \
      ghcr.io/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME:latest
    ```
    *(Note: The `//$(pwd)/...` syntax is recommended for Git Bash/MINGW64 on Windows. Use appropriate volume mounting syntax for your shell if different, e.g., `${PWD}` for PowerShell or just `./` for Linux/macOS)*

4.  **Check Output:**
    The reconstructed video (`reconstructed_jumbled_video.mp4`) will appear in your local `output_videos` folder.

---

## ⚙️ Algorithm & Architecture

### 1. Algorithm: CNN Features + TSP Optimization
* **Frame Extraction:** OpenCV reads the input video into 300 frames.
* **Feature Extraction:** A pre-trained ResNet-50 model (executed via the `ort` crate) processes all frames **sequentially** (for thread safety in this implementation) converting each into a high-dimensional feature vector representing its semantic content.
* **Distance Matrix:** A $300 \times 300$ matrix is computed using the **Cosine Distance** between all pairs of feature vectors.
* **Sequence Solving:** The distance matrix is treated as a Traveling Salesperson Problem. The highly optimized **LKH-3 heuristic** (via the `elkai-rs` crate) finds the shortest cycle connecting all frames.
* **Path Creation & Orientation:** The solved cycle is converted into a linear path by identifying and cutting the "longest link" (most dissimilar adjacent frames). A robust heuristic then checks the original frame indices along the path to ensure the correct temporal direction (forward vs. backward) before final output. Minor local path smoothing is also applied.
* **Video Writing:** OpenCV writes the re-ordered frames into the final output video file.

### 2. Architecture: Dockerized Native Binary
* **Performance:** The core logic is implemented in **Rust** and compiled to a native, optimized binary for maximum speed, eliminating Python overhead. Runtime is typically **< 30 seconds** on the benchmark CPU (dominated by sequential feature extraction).
* **Reproducibility:** The complex build environment (C++, CMake, OpenCV, specific Rust toolchain, ONNX Runtime library) is encapsulated within a **multi-stage Dockerfile**. This guarantees that the application builds and runs identically everywhere Docker is installed.
* **Deployment:** The final Docker image is pushed to GHCR, allowing judges (and others) to run the optimized solution with a single `docker run` command, requiring only Docker Desktop as a prerequisite. This demonstrates best practices in software deployment.

---

## (Optional) Building the Docker Image Locally

If you wish to build the Docker image yourself (e.g., after modifying the code), ensure Docker Desktop is running and execute:

```bash
# This will take 15-30+ minutes the first time
docker build -t frame-reconstructor .