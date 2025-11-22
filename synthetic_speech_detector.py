import librosa
import numpy as np
import joblib

class SyntheticSpeechDetector:
    def __init__(self, model_path, scaler_path):
        """Initialize with trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Verify model
        assert list(self.model.classes_) == [0, 1], "Model classes must be [0, 1]"
        print(f"✓ Model loaded - classes: {self.model.classes_}")
    
    def extract_features(self, audio_path, sr=16000):
        """Extract 67-dimensional feature vector"""
        try:
            y, sr = librosa.load(audio_path, sr=sr, duration=4.0)
            y_trim, _ = librosa.effects.trim(y, top_db=20)
            
            target_length = 4 * sr
            if len(y_trim) < target_length:
                y_trim = np.pad(y_trim, (0, target_length - len(y_trim)))
            else:
                y_trim = y_trim[:target_length]
            
            if np.max(np.abs(y_trim)) > 0:
                y_trim = y_trim / np.max(np.abs(y_trim))
            
            features = []
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral
            features.append(np.mean(librosa.feature.spectral_centroid(y=y_trim, sr=sr)))
            features.append(np.std(librosa.feature.spectral_centroid(y=y_trim, sr=sr)))
            features.append(np.mean(librosa.feature.spectral_rolloff(y=y_trim, sr=sr)))
            features.append(np.std(librosa.feature.spectral_rolloff(y=y_trim, sr=sr)))
            features.append(np.mean(librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)))
            features.append(np.std(librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)))
            
            # ZCR
            zcr = librosa.feature.zero_crossing_rate(y_trim)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y_trim, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y_trim, sr=sr)
            features.extend(np.mean(contrast, axis=1))
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            return None
    
    def detect(self, audio_path):
        """Detect if audio is real (bonafide) or fake (spoof)"""
        try:
            features = self.extract_features(audio_path)
            if features is None:
                return {'error': 'Failed to extract features'}
            
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # probabilities[0] = bonafide (class 0)
            # probabilities[1] = spoof (class 1)
            
            return {
                'prediction': 'Bonafide' if prediction == 0 else 'Synthetic',
                'confidence': float(max(probabilities)),
                'bonafide_probability': float(probabilities[0]),
                'synthetic_probability': float(probabilities[1])
            }
        except Exception as e:
            return {'error': str(e)}
