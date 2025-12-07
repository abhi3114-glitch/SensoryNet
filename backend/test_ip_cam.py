import cv2
import time

urls = [
    "http://192.168.29.197:4747/video",
    "http://192.168.29.197:4747/video/mjpg", # Another variant
    "http://192.168.29.197:4747/mjpegfeed"
]

print(f"Testing {len(urls)} DroidCam URLs with CAP_FFMPEG...")

for url in urls:
    print(f"\nTesting: {url}")
    # Force FFMPEG backend which handles network streams better
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"SUCCESS! Resolution: {w}x{h}")
            cap.release()
            break
        else:
            print("Opened but failed to read frame.")
    else:
        print("Failed to open.")
    cap.release()

print("SUCCESS: Opened URL")
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Resolution: {w}x{h}")
    print(f"Brightness: {gray.mean():.1f}")
else:
    print("Opened but failed to read frame")

cap.release()
