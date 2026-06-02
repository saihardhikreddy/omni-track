# 👁️ Omni-Track Command Center v12.0
**Real-Time Crowd Topography, SVD Anomaly Detection, and Fully-Connected Mesh Analysis.**

Omni-Track is an advanced Computer Vision and Linear Algebra pipeline designed for high-density crowd management. By layering Singular Value Decomposition (SVD) and Euclidean Graph Theory over standard YOLO/ByteTrack neural networks, this system translates raw pixel data into mathematical crowd topology.

---

## 🧠 How It Works: The Mathematics

Standard object detection models are "black boxes"—they identify *where* people are, but they do not understand the physical structure or flow of the crowd. Omni-Track solves this by bridging Deep Learning with pure Linear Algebra.

### 1. Dual AI Engines (The Vision)
The system utilizes two distinct AI brains running on NVIDIA GPU acceleration:
* **Live Mode:** Uses `YOLOv8-Large` coupled with ByteTrack for high-speed, temporal tracking of moving crowds.
* **Static Mode:** Uses Slicing Aided Hyper Inference (**SAHI**) to chop massive, ultra-high-resolution stadium images into smaller grids, finding micro-targets (distant faces) that standard convolutional networks miss.

### 2. SVD Rank-1 Topography (The Macro-Structure)
To understand the crowd's shape, the system maps live camera coordinates into a topological density matrix, $M$. 
* The system performs real-time **Singular Value Decomposition** ($M = U \Sigma V^T$).
* By extracting only the primary singular value ($\sigma_1$), the system computes a **Rank-1 Approximation**. This mathematically acts as a noise filter, isolating the "Core Structural Mass" of the crowd and mapping it dynamically with a **Blue Grid**.
* **Anomaly Detection:** Any individual who mathematically falls outside of this primary singular component (a residual data point) is logically isolated as a structural anomaly. The system flags them instantly with a **Red Grid**.

### 3. Fully Connected $K_n$ Network Graph (The Micro-Structure)
To calculate physical proximity and crush-risk tension, the system treats every detected person as a node in a fully connected graph.
* It computes the Euclidean hypotenuse between *every single active target* simultaneously.
* Using combinatorics, it visualizes the exact $O(N^2)$ computational complexity, calculating and drawing $\frac{N(N-1)}{2}$ connection vectors per frame (the **Yellow Web**).

---

## 🚀 Step-by-Step Installation

Follow these instructions to deploy Omni-Track on your local machine.

### Prerequisites
* **Python 3.9+** installed on your system.
* **Git** installed on your system.
* (Optional but highly recommended) An **NVIDIA GPU** for real-time live feed processing.

### 1. Clone the Repository
Open your terminal and download the code:
```bash
git clone [https://github.com/your-username/omni-track.git](https://github.com/your-username/omni-track.git)
cd omni-track
