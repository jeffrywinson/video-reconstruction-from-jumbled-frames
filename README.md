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

    ```bash
    # Make sure Docker Desktop is running!
    docker run --rm \
      -v "//$(pwd)/test_videos":/app/test_videos \
      -v "//$(pwd)/output_videos":/app/output_videos \
      ghcr.io/jeffrywinson/video-reconstruction-from-jumbled-frames:latest
    ```
    *(Note: The `//$(pwd)/...` syntax is recommended for Git Bash/MINGW64 on Windows. Use appropriate volume mounting syntax for your shell if different, e.g., `${PWD}` for PowerShell or just `./` for Linux/macOS)*

    ```powershell
    # Make sure Docker Desktop is running!
    docker run --rm -v "${PWD}\test_videos:/app/test_videos" -v "${PWD}\output_videos:/app/output_videos" ghcr.io/jeffrywinson/video-reconstruction-from-jumbled-frames:latest
    ```

    ```macOS/Linus (bash/zsh/sh)
    # Make sure Docker Desktop is running!
    docker run --rm -v "$(pwd)/test_videos:/app/test_videos" -v "$(pwd)/output_videos:/app/output_videos" ghcr.io/jeffrywinson/video-reconstruction-from-jumbled-frames:latest
    ```

4.  **Check Output:**
    The reconstructed video (`reconstructed_jumbled_video.mp4`) will appear in your local `output_videos` folder.

---

# 🧩 Algorithm Methodology & Design Rationale

Our high-level strategy is to **transform the visual sequencing problem into a data-driven graph problem.** We treat each frame as a "node" and use a ML model to find the "distance" between them. The problem then becomes finding the shortest path that visits every node once—a classic **Traveling Salesperson Problem (TSP)**.

---

## ⚙️ Core Dependencies & Rationale

* **`opencv`**: The industry standard for high-performance video I/O (reading/writing frames).
* **`ort` (ONNX Runtime)**: Provides a high-speed, Python-free native Rust interface for executing the ResNet-50 model.
* **`ndarray`**: The "NumPy of Rust," essential for all tensor and matrix operations, especially stacking the 4D batch tensor.
* **`elkai-rs`**: A Rust wrapper for **LKH-3**, the state-of-the-art heuristic solver for finding near-perfect TSP solutions in milliseconds.
* **`anyhow`**: Simplifies error handling across different libraries.

---

## 🔬 Step-by-Step Algorithmic Pipeline

1.  **Frame Extraction (`video_io.rs`)**: OpenCV reads the `.mp4` file into a `Vec<Mat>` (a list of 300 in-memory frames), saving the original FPS and resolution.

2.  **Image Preprocessing (`features.rs`)**: Each frame is converted from BGR to RGB, resized to $224 \times 224$, and normalized using ImageNet statistics. The result is a `[3, 224, 224]` tensor.

3.  **Batch Feature Extraction (`features.rs`)**: This is a key optimization. Instead of running the model 300 times, we **stack** all 300 preprocessed tensors into a *single* `[300, 3, 224, 224]` batch. This batch is fed to the ONNX Runtime *once*, which efficiently returns a `[300, 2048]` feature matrix, where each row is a "semantic fingerprint" of a frame.

4.  **Pairwise Distance Matrix (`solver.rs`)**: We compute the **Cosine Distance** between every unique pair of feature vectors. This creates a $300 \times 300$ symmetric matrix where `Matrix[i,j]` stores the *dissimilarity* between Frame $i$ and Frame $j$.

5.  **TSP Transformation & Solving (`solver.rs`)**:
    * **Dummy Node:** To find the shortest *path* (not cycle), we create a $301 \times 301$ matrix. We add a "dummy node" (index 300) with a distance of **0.0** to all other nodes.
    * **Solving:** We feed this $301 \times 301$ matrix to the **LKH-3 solver**. Because travel to/from the dummy node is "free," the optimal *cycle* it finds *must* use the dummy node to bridge the true start and end of the path (e.g., `[... Frame_End, Dummy_Node, Frame_Start, ...]`).
    * **Path Extraction:** We find the dummy node in the solved cycle, remove it, and rotate the list to get our ordered path.
    * **Orientation:** We run a quick heuristic to ensure the path isn't playing in reverse, and flip it if needed.

6.  **Video Re-assembly (`video_io.rs`)**: A new `VideoWriter` is created. We iterate through our `solved_order` vector and write the frames in the correct sequence to the new video file.

---

## 🤔 Design Rationale (The "Why")

### Why a CNN (ResNet-50) over Pixel-based Metrics?
* **Problem:** Raw pixel differences (MSE, etc.) are brittle. A slight camera pan or lighting change makes a frame look completely different, confusing the algorithm.
* **Our Choice:** A **CNN** understands *semantic content* (objects, textures). Consecutive frames have very similar semantic "fingerprints," making them "close" in the feature space even if the pixels are different.

### Why Batch Inference over Sequential?
* **Problem:** Calling the model 300 times (once per frame) is extremely slow due to high initialization overhead for each call.
* **Our Choice:** We **stack all 300 frames** into a single tensor and run the model *once*. This one-time cost allows the GPU/CPU to process the entire set in parallel, reducing a multi-minute step to seconds.

### Why Cosine Distance over Euclidean?
* **Problem:** Euclidean ($L_2$) distance is sensitive to vector *magnitude* (length) in high dimensions.
* **Our Choice:** **Cosine Distance** measures the *angle* (direction) between vectors. It is the standard for comparing semantic embeddings, as it purely measures content similarity.

### Why "Dummy Node" TSP over "Cut the Longest Link"?
* **Problem:** Solving a $300 \times 300$ cycle and "cutting" the longest link is a weak heuristic that can easily fail (e.g., if there's a hard cut in the video).
* **Our Choice:** The **"Dummy Node"** technique is the mathematically sound way to convert a path problem into a cycle problem. We let the LKH-3 solver *itself* find the optimal start and end points, which is far more robust.

### Why LKH-3 (TSP) over a Greedy Algorithm?
* **Problem:** A "nearest-neighbor" greedy algorithm makes locally optimal choices that lead to globally terrible solutions. One wrong jump early on derails the entire path.
* **Our Choice:** **LKH-3** performs a global optimization. It considers the *total cost of the entire path*, allowing it to find a near-perfect solution that a greedy algorithm would miss.

---

## 🚀 Future Scope & Potential Improvements

This implementation can be made even faster. The frame preprocessing (in `extract_batch`) and the distance matrix calculation (in `build_distance_matrix`) are both "embarrassingly parallel." Both of these `for` loops could be parallelized using the **`rayon`** crate (e.g., `par_iter()`) to leverage all available CPU cores, which would significantly reduce CPU-bound processing time.

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