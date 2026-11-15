import argparse
import cv2

import sys

def camera_backend(platform=None):
    """DirectShow on Windows (fast to open); elsewhere let OpenCV pick (V4L2, AVFoundation)."""
    platform = sys.platform if platform is None else platform
    return cv2.CAP_DSHOW if platform.startswith("win") else cv2.CAP_ANY

WINDOW = "Hardware Diagnostic"


def window_closed(name):
    """True once the window has been closed with its X button.

    waitKey() never sees a key after that, and the next imshow() opens the window
    again, so without this check the feed could only be stopped with Ctrl+C."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # some backends raise once the window is gone
        return True

def open_first_working(indices, backend=None, log=print):
    """(capture, index) for the first camera in indices that opens AND delivers a frame, or
    (None, -1). A camera that opens but streams nothing (OBS's virtual camera before
    "Start Virtual Camera", a webcam held by another app) is released and skipped."""
    backend = camera_backend() if backend is None else backend
    for idx in indices:
        log(f"[*] Probing index {idx}...")
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                log(f"[SUCCESS] Live camera locked on index {idx}!\n")
                return cap, idx
            log(f"[WARNING] Index {idx} opened, but no video data is streaming.")
        cap.release()
    return None, -1


def parse_args():
    parser = argparse.ArgumentParser(description="Omni-Track Camera Probe Diagnostic Utility")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Explicit camera index to test (e.g. 0, 1, 2)")
    parser.add_argument("--width", type=int, default=1920, help="Target frame width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Target frame height (default: 1080)")
    return parser.parse_args()

def main():
    """Probe the cameras and show a live feed. Returns 0 on success, 1 if no camera works."""
    args = parse_args()
    print("""
=========================================================
 📷 WEBCAM DIAGNOSTIC PROTOCOL
=========================================================
""")

    if args.camera is not None:
        probe_indices = [args.camera]
        print(f"[SYSTEM] Probing user-specified camera index {args.camera}...")
    else:
        probe_indices = [0, 1, 2, 3]
        print("[SYSTEM] Probing available camera feeds (indices 0, 1, 2, 3)...")

    cap, working_index = open_first_working(probe_indices)

    if not cap or not cap.isOpened():
        print("[ERROR] ❌ No working camera stream detected.")
        return 1

    print(f"[SYSTEM] Requesting resolution: {args.width}x{args.height}...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM] Stream initialized at: {int(actual_w)}x{int(actual_h)}")
    print("[SYSTEM] 🟢 FEED ACTIVE. Press 'q' or Esc, or close the video window, to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"CAMERA ACTIVE | INDEX: {working_index}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
        if window_closed(WINDOW):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())