# SensoryNet - Ambient Intelligence System

SensoryNet is a privacy-first, local-processing application designed to infer the ambient environment state (e.g., Quiet, Noisy, Conversation) using your laptop's built-in sensors. It processes microphone and camera data locally to provide real-time recommendations without ever sending raw audio or video data to the cloud.

## Features

- **Privacy-First Architecture**: All sensor data is processed in-memory and discarded immediately. No audio or video is ever saved to disk or transmitted.
- **Real-Time Environment Classification**: Uses a lightweight custom classifier to detect states like "Quiet", "Noisy", "Conversation", and "Dark".
- **Live Dashboard**: A modern Next.js frontend visualizing real-time metrics for:
  - **Audio**: RMS Amplitude, Decibel Levels, Frequency Band Energy (Low/Mid/High).
  - **Video**: Brightness, Motion Magnitude, Motion Hotspots.
- **Smart Sensor Management**:
  - **Auto-Switching Audio**: Prioritizes hardware microphones (Realtek, Arrays) over virtual drivers (Camo, DroidCam) to ensure active audio capture.
  - **Smart Camera Support**: Auto-detects available cameras, supporting built-in webcams, DroidCam, and Camo. Includes a robust "Mock Mode" fallback for testing without sensors.
- **Resource Efficient**: Designed to run with minimal CPU usage on standard laptops.

## Technology Stack

### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI (High-performance Async API)
- **Computer Vision**: OpenCV (cv2) for optical flow and brightness analysis.
- **Audio Processing**: SoundDevice and Numpy for real-time FFT and spectral analysis.
- **Networking**: WebSockets for low-latency state streaming to the frontend.
- **Machine Learning**: Custom Numpy-based centroid classifier (optimized for Windows/Python 3.14 compatibility).

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Visualization**: Recharts for dynamic data plotting.
- **Icons**: Lucide React

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- Webcam and Microphone (optional, but recommended)

### 1. Backend Setup

Navigate to the backend directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Start the backend server:

```bash
python -m uvicorn app.main:app --port 8000 --reload
```

The backend API will be available at `http://localhost:8000`.

### 2. Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

Start the development server:

```bash
npm run dev
```

The application dashboard will be available at `http://localhost:3000`.

## Configuration

### Camera Source
The system attempts to automatically detect the best available camera (Index 0, 1, 2, or 3). It prioritizes external feeds like DroidCam/Camo properly. If no camera is found, it enters a "Mock Mode" to simulate data.

### Microphone Source
The system automatically scans for hardware microphones (e.g., "Realtek Audio", "Microphone Array") and ignores virtual audio drivers that are often silent.

## Troubleshooting

- **Static Camera Image**: If using DroidCam, ensure the PC Client is running and you have clicked "Start". The backend may show "Brightness 80" if the driver is active but disconnected.
- **No Audio**: Ensure your microphone privacy settings allow desktop apps to access the microphone. The logs will confirm which device index is selected.
- **Backend Crashes**: Ensure you are using the specific versions listed in `requirements.txt`, especially for `numpy`, to avoid ABI incompatibilities on Windows.

## License

This project is open resource for educational and personal use.
