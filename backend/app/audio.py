import sounddevice as sd
import numpy as np
import threading
import time

# Constants
SAMPLE_RATE = 44100
BLOCK_SIZE = 4096

class AudioSensor:
    def __init__(self):
        self.stream = None
        self.running = False
        self.latest_features = {
            "rms": 0.0,
            "db": 0.0,
            "low_energy": 0.0,
            "mid_energy": 0.0,
            "high_energy": 0.0
        }
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _process_loop(self):
        # Auto-select input device
        try:
            devices = sd.query_devices()
            device_idx = None
            
            # 1. Prefer "Realtek" or "Microphone Array" (Built-in)
            # Skip "Camo" or "DroidCam" as they might be silent virtual drivers
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = dev['name']
                    # Skip virtual cams for audio
                    if "Camo" in name or "DroidCam" in name:
                        continue
                        
                    if "Realtek" in name or "Array" in name or "Internal" in name:
                        device_idx = i
                        print(f"AudioSensor: Found preferred hardware mic: {name} (Index {i})")
                        break
            
            # 1b. If no specific hardware mic found, try any "Microphone" but still skip virtuals if possible
            if device_idx is None:
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        name = dev['name']
                        if "Camo" in name or "DroidCam" in name:
                            continue
                        if "Microphone" in name or "Mic" in name:
                            device_idx = i
                            print(f"AudioSensor: Found generic mic: {name} (Index {i})")
                            break
            
            # 2. Fallback to any input
            if device_idx is None:
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        device_idx = i
                        print(f"AudioSensor: Fallback to input: {dev['name']} (Index {i})")
                        break
            
            if device_idx is None:
                device_idx = sd.default.device[0]
                print(f"AudioSensor: Using default device index {device_idx}")

            try:
                with sd.InputStream(device=device_idx, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=self._audio_callback):
                    while self.running:
                        sd.sleep(100)
            except Exception as e:
                print(f"AudioSensor: Stream failed ({e}). Switching to Mock Mode.")
                self._run_mock_loop()
                
        except Exception as e:
            print(f"AudioSensor Init Error: {e}. Switching to Mock Mode.")
            self._run_mock_loop()

    def _run_mock_loop(self):
        print("AudioSensor: Running MOCK DATA loop for demo.")
        while self.running:
            # Simulate "Quiet Room" with occasional speech
            t = time.time()
            
            # Base ambient noise (Low)
            rms = 0.001 + 0.0005 * np.random.rand()
            
            # Occasional speech spike (every 5-10 seconds)
            if int(t) % 8 == 0:
                rms += 0.05 + 0.02 * np.random.rand()  # Speaking volume
            
            db = 20 * np.log10(rms) if rms > 0 else -60
            db = max(-60, db + 60) # Normalize roughly to 0-60+ range for display
            
            # Frequencies
            low = rms * 0.5
            mid = rms * 0.3
            high = rms * 0.1
            
            with self.lock:
                self.latest_features = {
                    "rms": float(rms),
                    "db": float(db),
                    "low_energy": float(low),
                    "mid_energy": float(mid),
                    "high_energy": float(high)
                }
            time.sleep(0.1)


    def _audio_callback(self, indata, frames, time_info, status):
        # indata is numpy array (frames, channels)
        if status:
            print(f"Audio status: {status}")
        self._compute_features(indata[:, 0])

    def _compute_features(self, data):
        # Data is float32 normalized [-1, 1]
        
        # 1. RMS & dB
        rms = np.sqrt(np.mean(data**2))
        db = 20 * np.log10(rms) if rms > 1e-6 else -80
        
        # 2. FFT
        fft_spectrum = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0/SAMPLE_RATE)
        
        low_mask = freqs < 300
        mid_mask = (freqs >= 300) & (freqs < 2000)
        high_mask = freqs >= 2000
        
        low_energy = np.sum(fft_spectrum[low_mask])
        mid_energy = np.sum(fft_spectrum[mid_mask])
        high_energy = np.sum(fft_spectrum[high_mask])
        
        with self.lock:
            self.latest_features = {
                "rms": float(rms),
                "db": float(db),
                "low_energy": float(low_energy),
                "mid_energy": float(mid_energy),
                "high_energy": float(high_energy)
            }

    def get_features(self):
        with self.lock:
            return self.latest_features.copy()

if __name__ == "__main__":
    sensor = AudioSensor()
    sensor.start()
    try:
        while True:
            print(sensor.get_features())
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()
