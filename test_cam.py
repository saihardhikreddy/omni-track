import cv2
import time

print("""
=========================================================
 📷 WEBCAM DIAGNOSTIC PROTOCOL
=========================================================
""")
print("[SYSTEM] Hunting for USB/Kreo video feeds (Skipping built-in laptop cam)...")

cap = None
working_index = -1

# Bypassing 0 to avoid the integrated laptop camera trap
for idx in [1, 2, 3]:
    print(f"[*] Probing index {idx}...")
    temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) 
    
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
    print("[ERROR] ❌ No cameras found. Check USB connection and Kreo App.")
    exit()

print("[SYSTEM] Requesting 1080p resolution...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[SYSTEM] Stream initialized at: {int(actual_w)}x{int(actual_h)}")
print("[SYSTEM] 🟢 FEED ACTIVE. Press 'q' on the video window to close.")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"KREO CAMERA ACTIVE | INDEX: {working_index}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Hardware Diagnostic", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()