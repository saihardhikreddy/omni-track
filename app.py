import argparse

VERSION = "12.0.0"

def parse_cli():
    parser = argparse.ArgumentParser(
        prog="omni-track",
        description="OMNI-TRACK (SVD Topography + K_n Network Graph) v12.0 - High-density crowd monitoring system",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="Show program version and exit",
    )
    return parser.parse_known_args()

if __name__ == "__main__":
    parse_cli()

import sys

import cv2
import threading
import time
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from test_cam import camera_backend
import topography

def print_banner():
    print("""
    =========================================================
     ██████╗██████╗  ██████╗ ██╗    ██╗██████╗ 
    ██╔════╝██╔══██╗██╔═══██╗██║    ██║██╔══██╗
    ██║     ██████╔╝██║   ██║██║ █╗ ██║██║  ██║
    ██║     ██╔══██╗██║   ██║██║███╗██║██║  ██║
    ╚██████╗██║  ██║╚██████╔╝╚███╔███╔╝██████╔╝
     ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝ ╚═════╝ 
     OMNI-TRACK (SVD TOPOGRAPHY + K_n NETWORK GRAPH) v12.0
    =========================================================
    """)
    print("[SYSTEM] Waking up Dual Neural Engines...\n")

print_banner()

# --- 1. LOAD DUAL AI MODELS (GPU WHEN AVAILABLE) ---
def pick_device() -> str:
    """Use the NVIDIA GPU when there is one, otherwise run on the CPU (slower)."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = pick_device()
if DEVICE == "cpu":
    print("[WARNING] No CUDA GPU found, running on the CPU. Live tracking will be slow.")

try:
    print("[SYSTEM] Loading Engine A (YOLO-Large + ByteTrack) for Live Camera...")
    fast_model = YOLO('yolov8l.pt') 
    fast_model.to(DEVICE)

    print("[SYSTEM] Loading Engine B (SAHI-YOLO) for Static Images...")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path='best.pt',         
        confidence_threshold=0.15,    
        image_size=1024,              
        device=DEVICE,
    )
    print(f"[SUCCESS] Both AI Brains Loaded on {DEVICE}.\n")
except Exception as e:
    print(f"[ERROR] Failed to load models. {e}")
    sys.exit(1)

# ==========================================
# 📸 MODE 1: STATIC IMAGE (SAHI + DISK SAVE)
# ==========================================
def analyze_image():
    print("\n" + "="*40)
    print("[SYSTEM] STATIC IMAGE ANALYSIS MODE")
    print("="*40)
    print("[TIP] Drag and drop your image file directly into this terminal!")
    
    file_path = input("\nImage Path: ").strip(' "\'')
    
    if not file_path:
        print("[ERROR] No path provided.")
        return

    print(f"\n[SYSTEM] Executing Deep-Slice SAHI Analysis on {file_path}...")
    
    img = cv2.imread(file_path)
    if img is None: 
        print("[ERROR] Could not read the image. Check the file path.")
        return

    result = get_sliced_prediction(
        image=img, detection_model=sahi_model, slice_height=512, slice_width=512,
        overlap_height_ratio=0.35, overlap_width_ratio=0.35, perform_standard_pred=True,    
        postprocess_type="NMS", postprocess_match_threshold=0.3 
    )

    detections = result.object_prediction_list
    total = len(detections)

    max_dim = 1600
    h, w = img.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    if scale != 1.0: img = cv2.resize(img, (int(w * scale), int(h * scale)))

    for pred in detections:
        x1, y1 = int(pred.bbox.minx * scale), int(pred.bbox.miny * scale)
        x2, y2 = int(pred.bbox.maxx * scale), int(pred.bbox.maxy * scale)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.rectangle(img, (0, 0), (320, 60), (0, 0, 0), -1)
    cv2.putText(img, f"DEEP SCAN: {total} PEOPLE", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- TERMINAL OUTPUT & SAVE TO DISK ---
    print("\n" + "!"*40)
    print(f" 🎯 MISSION REPORT: {total} PEOPLE DETECTED")
    print("!"*40)
    
    output_filename = "scan_result.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\n[SUCCESS] Tactical image saved to your folder as '{output_filename}'")
    
    # Attempt to show window, gracefully fail if Windows blocks it
    try:
        cv2.imshow("Tactical Image Analysis (SAHI)", img)
        print("[SYSTEM] Press any key on the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("[WARNING] Window blocked by OS. Please open 'scan_result.jpg' manually from your folder.")

# ==========================================
# 🎥 MODE 2: LIVE CAMERA (OMNI-TRACK)
# ==========================================
latest_frame = None
latest_boxes = []
latest_total = 0
is_running = True

def fast_ai_thread():
    global latest_frame, latest_boxes, latest_total, is_running
    while is_running:
        if latest_frame is not None:
            frame_to_process = latest_frame.copy()
            results = fast_model.track(
                frame_to_process, conf=0.35, imgsz=1024, classes=[0], 
                persist=True, tracker="bytetrack.yaml", verbose=False
            )
            
            temp_boxes = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                for box in boxes:
                    temp_boxes.append((box[0], box[1], box[2], box[3]))
                
            latest_boxes = temp_boxes
            latest_total = len(temp_boxes)
        else:
            time.sleep(0.01)

def live_camera():
    global latest_frame, latest_boxes, latest_total, is_running
    is_running = True
    
    print("\n[SYSTEM] Booting Omni-Track (SVD Topography + Network Graph)...")
    ai_thread = threading.Thread(target=fast_ai_thread, daemon=True)
    ai_thread.start()

    cap = None
    # Hunts for external/virtual cameras first, falls back to laptop cam.
    # DirectShow exists only on Windows, so Linux and macOS use OpenCV's default.
    for idx in [1, 2, 3, 0]:
        temp_cap = cv2.VideoCapture(idx, camera_backend())
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap.release()

    if not cap or not cap.isOpened(): 
        print("[ERROR] No camera found.")
        is_running = False
        return
    
    cam_w, cam_h = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    # Mathematical Constants
    grid_cols, grid_rows = topography.GRID_COLS, topography.GRID_ROWS

    print("[SYSTEM] 🟢 LIVE TRACKING ACTIVE. Press 'q' on the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Camera stopped sending frames.")
            break
        
        frame = cv2.flip(frame, 1) 
        latest_frame = frame.copy() 
        display_frame = frame.copy()
        
        frame_h, frame_w = display_frame.shape[:2]

        # Density grid, rank-1 SVD core and the K_n mesh (see topography.py).
        points = topography.centroids(latest_boxes)
        for (x1, y1, x2, y2) in latest_boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        M = topography.density_grid(points, frame_w, frame_h, grid_cols, grid_rows)

        # --- THE FULLY CONNECTED MESH MATH ---
        edges = topography.mesh_edges(points)
        network_connections = len(edges)
        for p1, p2 in edges:
            # Draw the connecting line automatically (No distance check!)
            cv2.line(display_frame, p1, p2, (0, 255, 255), 1)
        # One dot per person, drawn over the lines. (Dots used to be drawn for both ends of
        # every edge: n(n-1) circles, about 10,000 a frame for 100 people, and none at all
        # for a single person.)
        for p in points:
            cv2.circle(display_frame, p, 4, (0, 255, 255), -1)

        # --- THE SVD TOPOGRAPHY MATH ---
        core_cells, anomaly_cells = topography.classify_cells(M)
        anomalies_detected = len(anomaly_cells)
        for kind, cells in (("core", core_cells), ("anomaly", anomaly_cells)):
            for row, col in cells:
                px1 = int((col / grid_cols) * frame_w)
                py1 = int((row / grid_rows) * frame_h)
                px2 = int(((col + 1) / grid_cols) * frame_w)
                py2 = int(((row + 1) / grid_rows) * frame_h)
                if kind == "core":
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (255, 200, 0), 2)
                else:
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                    cv2.line(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 1)
                    cv2.line(display_frame, (px2, py1), (px1, py2), (0, 0, 255), 1)
                    cv2.putText(display_frame, "ANOMALY", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Dynamic UI Update Panel
        panel_height = 110 if (anomalies_detected > 0 or network_connections > 0) else 50
        cv2.rectangle(display_frame, (0, 0), (380, panel_height), (0, 0, 0), -1)
        
        cv2.putText(display_frame, f"LIVE COUNT: {latest_total}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset = 70
        if anomalies_detected > 0:
            cv2.putText(display_frame, f"SVD ANOMALIES: {anomalies_detected}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
        if network_connections > 0:
            cv2.putText(display_frame, f"NETWORK CONNECTIONS: {network_connections}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Omni-Track Command Center", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    is_running = False  # also stop the AI thread when the camera drops out
    latest_frame = None  # don't re-track a stale frame on the next run
    cap.release()
    cv2.destroyAllWindows()
    ai_thread.join(timeout=1.0)
    print("[SYSTEM] Camera offline.")

# ==========================================
# MAIN MENU
# ==========================================
while True:
    print("\n=== SYSTEM MENU ===")
    print("1. Deep Scan Stadium Image (SAHI + Disk Save)")
    print("2. Start Live Camera Feed (SVD + Network Graph)")
    print("3. Exit")
    
    choice = input("Enter choice (1/2/3): ")
    if choice == '1': analyze_image()
    elif choice == '2': live_camera()
    elif choice == '3': 
        print("\n[SYSTEM] Shutting down. Good luck on the Viva!\n")
        break
    else:
        print("[ERROR] Invalid selection. Try again.")