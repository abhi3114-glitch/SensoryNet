import cv2
import time

print("Testing camera indices...")

for idx in [0, 1, 2, 3]:
    print(f"\n--- Testing Index {idx} ---")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Index {idx}: Failed to open")
        continue
    
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        print(f"Index {idx}: SUCCESS. Resolution={w}x{h}, Brightness={brightness:.1f}")
    else:
        print(f"Index {idx}: Opened but failed to read frame")
    
    cap.release()
