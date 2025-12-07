import numpy as np
import pickle
import os
import json

class EnvironmentClassifier:
    def __init__(self, model_path="model.json"):
        self.model_path = model_path
        self.centroids = {}
        self.scaler_mean = None
        self.scaler_scale = None
        self.states = ["Quiet", "Noisy", "Crowded", "Silent", "Windy", "Conversation", "TV/Media", "Sleepy"]
        
        self.load_or_train_demo()

    def load_or_train_demo(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    data = json.load(f)
                    self.centroids = {k: np.array(v) for k, v in data["centroids"].items()}
                    self.scaler_mean = np.array(data["scaler_mean"])
                    self.scaler_scale = np.array(data["scaler_scale"])
                    print("Loaded existing model.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.train_demo()
        else:
            self.train_demo()

    def train_demo(self):
        print("Training demo model (Lightweight)...")
        # Features: [rms, db, low, mid, high, brightness, motion_mag, motion_hotspots]
        X = []
        y = []
        
        # 1. Quiet
        for _ in range(50):
            X.append([0.01, 20, 10, 5, 2, 100, 0, 0])
            y.append("Quiet")
            
        # 2. Noisy (High RMS, broad spectrum)
        for _ in range(50):
            X.append([0.5, 80, 500, 500, 500, 100, 5, 1])
            y.append("Noisy")
            
        # 3. Silent
        for _ in range(50):
            X.append([0.001, 10, 1, 0, 0, 100, 0, 0])
            y.append("Silent")

        # 4. Conversation (Mid band energy, varying RMS)
        for _ in range(50):
            X.append([0.2, 60, 100, 800, 100, 100, 2, 1])
            y.append("Conversation")

        # 5. Sleepy (Dark, Silent)
        for _ in range(50):
            X.append([0.005, 15, 2, 1, 0, 20, 0, 0])
            y.append("Sleepy")

        X = np.array(X)
        
        # Simple Standard Scaler
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_scale = np.std(X, axis=0)
        self.scaler_scale[self.scaler_scale == 0] = 1.0 # Avoid divide by zero
        
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        
        # Calculate centroids
        self.centroids = {}
        unique_classes = set(y)
        for cls in unique_classes:
            indices = [i for i, label in enumerate(y) if label == cls]
            class_samples = X_scaled[indices]
            self.centroids[cls] = np.mean(class_samples, axis=0)
        
        # Save as JSON (portable)
        data = {
            "centroids": {k: v.tolist() for k, v in self.centroids.items()},
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist()
        }
        with open(self.model_path, "w") as f:
            json.dump(data, f)
        print("Demo model trained and saved.")

    def predict(self, audio_feats, video_feats):
        # Combine features
        # [rms, db, low, mid, high, brightness, motion_mag, motion_hotspots]
        features = np.array([
            audio_feats.get("rms", 0),
            audio_feats.get("db", 0),
            audio_feats.get("low_energy", 0),
            audio_feats.get("mid_energy", 0),
            audio_feats.get("high_energy", 0),
            video_feats.get("brightness", 0),
            video_feats.get("motion_magnitude", 0),
            video_feats.get("motion_hotspots", 0)
        ])
        
        if self.scaler_mean is None:
            return "Unknown", 0.0

        features_scaled = (features - self.scaler_mean) / self.scaler_scale
        
        best_state = "Unknown"
        min_dist = float("inf")
        
        # Find nearest centroid
        distances = {}
        for state, centroid in self.centroids.items():
            dist = np.linalg.norm(features_scaled - centroid)
            distances[state] = dist
            if dist < min_dist:
                min_dist = dist
                best_state = state
        
        # Pseudo-confidence (1 / (1 + dist))
        confidence = 1.0 / (1.0 + min_dist)
        
        return best_state, confidence
