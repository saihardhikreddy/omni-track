# 👁️ Omni-Track Command Center
**Real-Time Crowd Topography, SVD Anomaly Detection, and Fully-Connected Mesh Analysis.**

Omni-Track is an advanced Computer Vision and Linear Algebra pipeline designed for high-density crowd management. By layering Singular Value Decomposition (SVD) and Euclidean Graph Theory over standard YOLO/ByteTrack neural networks, this system translates raw pixel data into mathematical crowd topology.

## 🚀 Key Features

* **SVD Rank-1 Topography:** Maps live camera coordinates into a topological matrix ($M = U \Sigma V^T$). By extracting the primary singular value, the system isolates the "Core Structural Mass" of a crowd (Blue Grid) and mathematically flags detached stragglers as anomalies (Red Grid).
* **Fully Connected $K_n$ Network Graph:** Computes real-time combinatorial distance vectors between all active targets using $\frac{N(N-1)}{2}$ complexity, visualizing spatial proximity and crush-risk tension in real-time.
* **Deep-Slice SAHI Scanning:** Integrates Slicing Aided Hyper Inference (SAHI) for ultra-high-resolution static imagery, finding micro-targets (distant faces) that standard convolutional networks miss.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/saihardhikreddy/omni-track.git](https://github.com/saihardhikreddy/omni-track.git)
   cd omni-track