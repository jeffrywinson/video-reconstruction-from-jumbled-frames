# Jumbled Frames Reconstruction Challenge (High-Performance Rust Solution) 🔥

This project reconstructs a 10-second, 30 FPS video from 300 randomly shuffled frames using a high-performance, native Rust application packaged in Docker.

The core algorithm leverages Convolutional Neural Networks (CNN) for semantic frame analysis and solves the frame ordering problem as a Traveling Salesperson Problem (TSP) using the highly optimized LKH-3 heuristic.

**Key Technologies:**
* **Language:** Rust 🦀
* **AI Model:** ResNet-50 (via ONNX Runtime / `ort` crate) 👁️
* **Video I/O:** OpenCV (`opencv-rust` crate) 🎥
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
    * It then **instantly runs** the compiled Rust program (< 25 seconds runtime expected on the benchmark i7).
    * It mounts your local `test_videos` and `output_videos` folders for input/output.

    Git Bash
    ```bash
    # Make sure Docker Desktop is running!
    docker run --rm \
      -v "//$(pwd)/test_videos":/app/test_videos \
      -v "//$(pwd)/output_videos":/app/output_videos \
      ghcr.io/jeffrywinson/video-reconstruction-from-jumbled-frames:latest
    ```
    *(Note: The `//$(pwd)/...` syntax is recommended for Git Bash/MINGW64 on Windows. Use appropriate volume mounting syntax for your shell if different, e.g., `${PWD}` for PowerShell or just `./` for Linux/macOS)*

    Powershell
    ```powershell
    # Make sure Docker Desktop is running!
    docker run --rm -v "${PWD}\test_videos:/app/test_videos" -v "${PWD}\output_videos:/app/output_videos" ghcr.io/jeffrywinson/video-reconstruction-from-jumbled-frames:latest
    ```

    macOS/Linus (bash/zsh/sh)
    ```bash
    # Make sure Docker Desktop is running!
    docker run --rm -v "$(pwd)/test_videos:/app/test_videos" -v "$(pwd)/output_videos:/app/output_videos" ghcr.io/jeffrywinson/video-reconstruction-from-jumbled-frames:latest
    ```

4.  **Check Output:**
    The reconstructed video (`reconstructed_jumbled_video.mp4`) will appear in your local `output_videos` folder.

---

# 🧩 Algorithm Methodology & Design Rationale

This document details the algorithmic approach and technical design choices for the "Jumbled Frame Reconstruction" project. The core problem is to re-sequence 300 shuffled video frames into their correct temporal order.

Our high-level strategy is to **transform a visual sequencing problem into a data-driven graph problem.** We treat each frame as a "node" in a graph and use a machine learning model to determine the "distance" (or dissimilarity) between every pair of nodes. The problem is then reduced to finding the single shortest path that visits every node exactly once—a classic computer science problem known as the **Traveling Salesperson Problem (TSP)**.

---

## ⚙️ Core Dependencies & Rationale

The selection of our core libraries was critical for achieving a balance of high performance, correctness, and deployment simplicity.

* **`opencv`**: Used for all video I/O (reading/writing frames).
    * **Why?** It is the battle-tested, industry-standard C++ library for computer vision, and its Rust bindings are robust. It provides the most efficient way to demux video files into raw frames and re-assemble them.
* **`ort` (ONNX Runtime)**: Used for running deep learning model inference.
    * **Why?** `ort` is a high-performance, cross-platform inference engine from Microsoft. It allows us to execute the pre-trained `resnet50.onnx` model from native Rust, completely eliminating Python overhead and achieving performance on par with a C++ application.
* **`ndarray`**: The "NumPy of Rust." Used for all tensor and matrix operations.
    * **Why?** It is essential for:
        1.  Converting preprocessed frames into a 3D tensor (`C, H, W`).
        2.  **Stacking** all 3D tensors into a 4D *batch* tensor (`B, C, H, W`).
        3.  Storing the $N \times N$ `distance_matrix`.
* **`elkai-rs`**: A Rust wrapper for the highly-optimized **LKH-3** TSP solver.
    * **Why?** LKH-3 is widely considered the *state-of-the-art heuristic solver* for the symmetric TSP. For a 300-node problem, it provides a near-perfect (and often truly optimal) solution in milliseconds. It is vastly superior to simpler, less accurate heuristics.
* **`anyhow`**: Used for simple and clean error handling.
    * **Why?** It simplifies the error-handling boilerplate, allowing us to use `?` on functions that return different error types, which is common when interfacing with C++ libraries (like OpenCV and ORT).

---

## 🔬 Step-by-Step Algorithmic Pipeline

Our pipeline, as seen in `main.rs`, executes five distinct steps.

### 1. Frame Extraction (`video_io.rs`)

The `extract_frames` function uses OpenCV to open the `jumbled_video.mp4` file. It iterates through the video, decodes every frame, and stores all 300 frames in memory as a `Vec<Mat>`, where `Mat` is OpenCV's primary image data structure. The original video's FPS and resolution are also saved.

### 2. Image Preprocessing (`features.rs`)

This is a critical preparatory step for the AI model. The `preprocess_frame` function takes a single `Mat` object and:
1.  **Converts Color:** OpenCV reads frames in **BGR** format. This is converted to standard **RGB**.
2.  **Resizes:** The frame is resized to $224 \times 224$ pixels, the required input size for ResNet-50.
3.  **Normalizes:** Pixel values (originally $0-255$) are scaled to $0.0-1.0$. Then, they are normalized using the standard ImageNet dataset's `MEAN` and `STD` values.
4.  **Creates Tensor:** The data is rearranged into an `ndarray::Array3` with the shape `[3, 224, 224]` (Channels, Height, Width).

### 3. Batch Feature Extraction (`features.rs`)

This is the core performance bottleneck and is heavily optimized. Instead of running the model 300 times, the `extract_batch` function:
1.  **Collects** all 300 preprocessed `Array3` tensors.
2.  **Stacks** them into a single, massive `ndarray::Array4` batch tensor with the shape `[300, 3, 224, 224]`.
3.  **Feeds** this entire batch to the ONNX Runtime (`session.run()`) in a *single call*.
4.  The ResNet-50 model processes this batch and returns a 2D feature matrix (e.g., `[300, 2048]`), where each row is a high-dimensional "semantic fingerprint" for the corresponding frame.

### 4. Pairwise Distance Matrix (`solver.rs`)

The `build_distance_matrix` function takes the `[300, 2048]` feature matrix. It iterates through every unique pair of feature vectors $(i, j)$ and computes their **Cosine Distance**. This produces a $300 \times 300$ symmetric matrix where `Matrix[i,j]` stores a floating-point value representing how *dissimilar* Frame $i$ and Frame $j$ are.

### 5. TSP Transformation & Solving (`solver.rs`)

This is the most complex algorithmic step. We need to find the shortest *path*, but TSP solvers find the shortest *cycle*.
1.  **"Dummy Node" Transformation:** We create an `(N+1) x (N+1)` matrix (i.e., $301 \times 301$). The original $300 \times 300$ matrix is copied into the top-left corner. All distances from the new "dummy node" (index 300) to all *other* nodes (and back) are set to **0.0**.
2.  **TSP Solving:** This new $301 \times 301$ matrix (scaled to integers) is fed to the **LKH-3 solver**. The solver finds the shortest *cycle* through all 301 nodes.
3.  **Path Extraction:** Because travel to/from the dummy node is "free," the optimal cycle will *always* include the segment `[..., Frame_End, Dummy_Node, Frame_Start, ...]`. We find the dummy node's position in the solved cycle, "un-rotate" the cycle, and remove the dummy node. This leaves us with the perfect, ordered *path* (e.g., `[Frame_Start, ..., Frame_End]`).
4.  **Path Orientation:** The path we extracted could be forwards or backwards. To fix this, we *first* find the single most dissimilar pair of frames in the *entire* dataset (our `start_candidate` and `end_candidate`). We then check our extracted path's endpoints and reverse the path *only if* the reversed orientation is a better match for our heuristic.

### 6. Video Re-assembly (`video_io.rs`)

The `save_video` function creates a new OpenCV `VideoWriter`. It then iterates through the `solved_order` `Vec<usize>` and, for each index, writes the corresponding frame from the in-memory `Vec<Mat>` to the new video file, using the original FPS and resolution.

---

## 🤔 Design Rationale (The "Why")

This section explains *why* these specific architectural and algorithmic choices were made over simpler alternatives.

### Why a CNN (ResNet-50) over Pixel-based Metrics?

* **Alternative:** Compare frames using raw pixel differences (e.g., Mean Squared Error or L1 distance).
* **Problem:** This method is extremely brittle. It fails completely if the camera pans slightly, an object moves, or lighting changes. A 1-pixel shift can result in a massive distance, making it impossible to distinguish a true adjacent frame from a random one.
* **Our Choice:** A pre-trained **Convolutional Neural Network (CNN)** like ResNet-50 does not compare pixels; it compares *semantic content*. It has been trained on millions of images and understands high-level features like "tree," "face," "building," etc. Consecutive frames in a video have *very* similar semantic feature vectors, even if the pixels themselves are different. This makes the "distance" between true frames $(N, N+1)$ exceptionally small, while the distance between non-adjacent frames is large.

### Why Batch Inference over Sequential?

* **Alternative:** Loop 300 times, calling `session.run()` for each frame individually.
* **Problem:** This is *extremely* slow. There is a high fixed cost to initializing an inference run and moving data to the GPU/CPU. Doing this 300 times would be the dominant bottleneck, likely taking minutes.
* **Our Choice:** As seen in `features.rs`, we stack all 300 frames into one `[300, 3, 224, 224]` tensor. This pays the inference startup cost *only once*. The GPU can then process the entire batch in a highly parallel, optimized fashion. This is the single most important performance optimization in the pipeline, reducing a multi-minute step to mere seconds.

### Why Cosine Distance?

* **Alternative:** Euclidean Distance ($L_2$ norm).
* **Problem:** Euclidean distance measures the "straight-line" distance between two vector endpoints. In high-dimensional spaces, this is sensitive to the *magnitude* (length) of the vectors, not just their content.
* **Our Choice:** **Cosine Distance** measures the *angle* between the two feature vectors. It is the standard for comparing high-dimensional embeddings (used in text, images, etc.) because it purely measures *similarity of content* (i.e., "direction") regardless of vector magnitude.

### Why "Dummy Node" TSP over "Cut the Longest Link"?

* **Alternative:** Solve a standard $300 \times 300$ TSP cycle, find the "longest link" (most dissimilar adjacent pair) in the solution, and "cut" the cycle there to create a path.
* **Problem:** This is a *heuristic on top of a heuristic*. It *assumes* the longest link in the solved cycle corresponds to the video's start/end. This assumption can be wrong (e.g., in a scene with a hard cut).
* **Our Choice:** The **"Dummy Node"** technique (as implemented in `solver.rs`) is the canonical, mathematically sound method for transforming a shortest *path* problem into a shortest *cycle* problem. By adding a node with 0-cost edges, we *let the LKH-3 solver itself* find the optimal start and end points of the path, as those will be the two nodes that connect to the dummy node. This is more robust, elegant, and correct.

### Why LKH-3 (TSP) over a Greedy Algorithm?

* **Alternative:** A simple "nearest-neighbor" greedy algorithm: Start at frame 0. Find the *closest* unvisited frame. Jump to it. From there, find the *next* closest unvisited frame, and so on.
* **Problem:** This is fast $(O(N^2))$ but highly prone to errors. It makes locally optimal choices that lead to a globally terrible solution. One "wrong jump" early on can trap the algorithm in a completely incorrect path with no way to backtrack.
* **Our Choice:** A **global optimization heuristic (LKH-3)** considers the *total cost of the entire path*. It is free to make a "locally bad" jump (e.g., skipping a close frame) if that jump enables a much better path overall. This global perspective is essential for achieving high accuracy and is why we accept the (still very fast) cost of a full TSP solve.

---

## 🚀 Future Scope & Potential Improvements

While this architecture is robust and performant, several areas could be parallelized to further improve speed.

* **Parallel Preprocessing:** The `extract_batch` function currently preprocesses all 300 frames sequentially on the main thread. This entire mapping operation (BGR->RGB, resize, normalize) is "embarrassingly parallel." This could be easily parallelized using the **`rayon`** crate, converting `frames.iter().map(...)` to `frames.par_iter().map(...)` to leverage all available CPU cores.
* **Parallel Distance Matrix:** The `build_distance_matrix` function uses two nested loops $(O(N^2))$. The outer loop (`for i in 0..n`) is also "embarrassingly parallel." `rayon` could be used to compute multiple rows of the distance matrix in parallel, as each row `i` is independent of all other rows.
* **Hardware-Specific Runtimes:** The `ort` crate can be configured to use hardware-specific **Execution Providers** (e.g., NVIDIA's TensorRT or Apple's CoreML). Compiling the application with these providers enabled could grant further, "free" performance boosts on compatible hardware by using highly optimized device-specific kernels.

---

## (Optional) Building the Docker Image Locally

If you wish to build the Docker image yourself (e.g., after modifying the code), ensure Docker Desktop is running and execute:

```bash
# This will take 15-30+ minutes the first time
docker build -t frame-reconstructor .
```

Modified the Dockerfile? Run:

```bash
docker build --no-cache -t frame-reconstructor .
```

Then you can proceed with the docker run commands already mentioned above.
