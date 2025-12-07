import cv2
import numpy as np
import threading
import time

class VideoSensor:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.latest_features = {
            "brightness": 0,
            "motion_magnitude": 0,
            "motion_hotspots": 0
        }
        self.lock = threading.Lock()
        self.prev_gray = None

    def start(self):
        if self.running:
            return
        self.running = True
        
        # Priority search order: 1 (DroidCam), 0 (Built-in), 2, 3
        # Users can hardcode an IP here if needed, e.g. "http://192.168.1.100:4747/video"
        search_order = [1, 0, 2, 3]
        
        # Override for DroidCam IP if needed (User can edit this or we can expose it via API)
        # self.camera_index can be set to a string
        if isinstance(self.camera_index, str):
             search_order = [self.camera_index] + search_order

        found = False
        print(f"VideoSensor: Starting camera search with order {search_order}")
        
        for idx in search_order:
            print(f"VideoSensor: Trying camera source {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.camera_index = idx
                    print(f"VideoSensor: Successfully connected to camera source {idx}")
                    found = True
                    break
                else:
                    print(f"VideoSensor: Camera source {idx} opened but failed to read frame.")
                    cap.release()
            else:
                print(f"VideoSensor: Failed to open camera source {idx}")
        
        if not found:
             print("VideoSensor: No working camera found. Starting dummy loop.")
        
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _generate_mock_data(self):
        # Generate data for "Good Lighting, Little Motion" (User Demo Scenario)
        t = time.time()
        
        # Brightness: High and stable (Good lighting)
        # Range: ~190-210
        brightness = 200 + 10 * np.sin(t * 0.2) 
        
        # Motion: Mostly still, with very subtle movements (User sitting at laptop)
        # Occasional small movements (typing, breathing)
        base_motion = 1.0 + 0.5 * np.sin(t * 1.5)
        noise = max(0, 0.5 * np.random.randn())
        motion_mag = base_motion + noise
        
        hotspots = int(motion_mag * 0.5) 
        
        # Very rare larger movement (e.g. adjust position), but not chaotic
        if int(t) % 15 == 0:
            motion_mag += 5.0
            hotspots += 2

        # Debug print rarely
        if int(t) % 5 == 0 and int(t * 10) % 10 == 0:
             print(f"Video Mock: Brightness={brightness:.1f}, Motion={motion_mag:.1f} (Demo Mode)")

        with self.lock:
            self.latest_features = {
                "brightness": float(brightness),
                "motion_magnitude": float(motion_mag),
                "motion_hotspots": int(hotspots)
            }

    def _process_loop(self):
        # SET TO TRUE FOR MOCK DATA (User Request)
        FORCE_MOCK = False
        
        while self.running:
            if FORCE_MOCK or self.cap is None or not self.cap.isOpened():
                # Fallback: Process mock data if no camera is connected or forced
                self._generate_mock_data()
                time.sleep(0.1) # Simulate 10 FPS
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Resize for performance and privacy (discard detail)
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Brightness
            brightness = np.mean(gray)
            
            # 2. Optical Flow (Motion)
            motion_mag = 0
            hotspots = 0
            
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mag = np.mean(mag)
                hotspots = np.sum(mag > 2.0) # Count pixels with significant motion
            
            self.prev_gray = gray
            
            # Debug: Print stats periodically
            if int(time.time()) % 5 == 0:
                 print(f"Video Debug: Brightness={brightness:.1f}, Motion={motion_mag:.1f}")

            with self.lock:
                self.latest_features = {
                    "brightness": float(brightness),
                    "motion_magnitude": float(motion_mag),
                    "motion_hotspots": int(hotspots)
                }
            
            # Sleep to reduce CPU usage (e.g., 5 FPS)
            time.sleep(0.2)

    def get_features(self):
        with self.lock:
            return self.latest_features.copy()

if __name__ == "__main__":
    sensor = VideoSensor()
    sensor.start()
    try:
        while True:
            print(sensor.get_features())
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()
