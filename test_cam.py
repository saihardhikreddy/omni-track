import argparse
import cv2
import time

import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Omni-Track Camera Probe Diagnostic Utility")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Explicit camera index to test (e.g. 0, 1, 2)")
    parser.add_argument("--width", type=int, default=1920, help="Target frame width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Target frame height (default: 1080)")
    return parser.parse_args()

def main():
    args = parse_args()
    print("""
=========================================================
 📷 WEBCAM DIAGNOSTIC PROTOCOL
=========================================================
""")

    cap = None
    working_index = -1
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

    if args.camera is not None:
        probe_indices = [args.camera]
        print(f"[SYSTEM] Probing user-specified camera index {args.camera}...")
    else:
        probe_indices = [0, 1, 2, 3]
        print("[SYSTEM] Probing available camera feeds (indices 0, 1, 2, 3)...")

    for idx in probe_indices:
        print(f"[*] Probing index {idx}...")
        temp_cap = cv2.VideoCapture(idx, backend)
        
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                cap = temp_cap
                working_index = idx
                print(f"[SUCCESS] Live camera locked on index {idx}!\n")
                break
            else:
                print(f"[WARNING] Index {idx} opened, but no video data is streaming.")
                temp_cap.release()

    if not cap or not cap.isOpened():
        print("[ERROR] ❌ No working camera stream detected.")
        return

    print(f"[SYSTEM] Requesting resolution: {args.width}x{args.height}...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM] Stream initialized at: {int(actual_w)}x{int(actual_h)}")
    print("[SYSTEM] 🟢 FEED ACTIVE. Press 'q' on the video window to close.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"CAMERA ACTIVE | INDEX: {working_index}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Hardware Diagnostic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()