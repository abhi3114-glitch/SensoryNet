from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .audio import AudioSensor
from .video import VideoSensor
from .ml import EnvironmentClassifier
import asyncio
import json
import time

app = FastAPI(title="SensoryNet Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
audio_sensor = AudioSensor()
# Auto-detect camera (Index 0 likely DroidCam if installed)
video_sensor = VideoSensor()
classifier = EnvironmentClassifier() # Trains on init if needed

# Recommendations
RECOMMENDATIONS = {
    "Quiet": "Perfect environment for deep work.",
    "Noisy": "Consider using noise-canceling headphones.",
    "Crowded": "Might be a good time for a coffee break if distracted.",
    "Silent": "It's very quiet. Maybe play some background music?",
    "Windy": "Check for fans or open windows causing noise.",
    "Conversation": "Meeting time? Or find a booth if you need focus.",
    "Sleepy": "Low light and silence detected. Turn on lights!",
    "TV/Media": "Media consumption detected."
}

@app.on_event("startup")
async def startup_event():
    # Auto-start sensors for demo (or make configurable)
    audio_sensor.start()
    video_sensor.start()

@app.on_event("shutdown")
async def shutdown_event():
    audio_sensor.stop()
    video_sensor.stop()

@app.get("/")
def read_root():
    return {"status": "SensoryNet Running"}

@app.get("/status")
def get_status():
    audio_feats = audio_sensor.get_features()
    video_feats = video_sensor.get_features()
    state, confidence = classifier.predict(audio_feats, video_feats)
    
    return {
        "state": state,
        "confidence": confidence,
        "recommendation": RECOMMENDATIONS.get(state, "No recommendation"),
        "features": {
            "audio": audio_feats,
            "video": video_feats
        },
        "sensors": {
            "audio": audio_sensor.running,
            "video": video_sensor.running
        }
    }

@app.post("/sensors/{sensor_type}/{action}")
def control_sensors(sensor_type: str, action: str):
    if sensor_type == "audio":
        if action == "start":
            audio_sensor.start()
        elif action == "stop":
            audio_sensor.stop()
    elif sensor_type == "video":
        if action == "start":
            video_sensor.start()
        elif action == "stop":
            video_sensor.stop()
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_feats = audio_sensor.get_features()
            video_feats = video_sensor.get_features()
            state, confidence = classifier.predict(audio_feats, video_feats)
            
            payload = {
                "timestamp": time.time(),
                "state": state,
                "confidence": confidence,
                "recommendation": RECOMMENDATIONS.get(state, ""),
                "features": {
                    "audio": audio_feats,
                    "video": video_feats
                }
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5) # Update rate
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WS Error: {e}")
        await websocket.close()
