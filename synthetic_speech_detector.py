
import numpy as np
import librosa
import joblib
import os

class SyntheticSpeechDetector:
    def __init__(self, model_path, scaler_path):
        """Initialize the detector with trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.sr = 16000
    
    def extract_features(self, audio_path):
        """Extract features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=4.0)
            
            # Preprocessing
            y_trim, _ = librosa.effects.trim(y, top_db=20)
            
            # Pad or truncate
            target_length = 4 * self.sr
            if len(y_trim) < target_length:
                y_trim = np.pad(y_trim, (0, target_length - len(y_trim)))
            else:
                y_trim = y_trim[:target_length]
            
            # Normalize
            y_trim = y_trim / (np.max(np.abs(y_trim)) + 1e-6)
            
            features = []
            
            # MFCC Features
            mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y_trim, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trim, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)
            
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth)
            ])
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y_trim)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=y_trim, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y_trim, sr=sr)
            features.extend(np.mean(contrast, axis=1))
            
            return np.array(features)
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    def detect(self, audio_path):
        """Detect if audio is synthetic or bonafide"""
        features = self.extract_features(audio_path)
        
        if features is None:
            return {"error": "Failed to extract features"}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0]
        
        result = {
            "prediction": "Bonafide" if prediction == 1 else "Synthetic",
            "confidence": float(max(confidence)),
            "bonafide_probability": float(confidence[1]),
            "synthetic_probability": float(confidence[0])
        }
        
        return result

# Usage example:
# detector = SyntheticSpeechDetector('xgboost_model.pkl', 'scaler.pkl')
# result = detector.detect('audio_file.wav')
# print(result)
